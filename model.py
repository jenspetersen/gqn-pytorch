import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from util import ConvModule


class ConvLSTMCell(ConvModule):

    _default_conv_kwargs = dict(
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1
    )

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_op=nn.Conv2d,
                 conv_kwargs=None,
                 concat_hidden_and_input=True,
                 **kwargs):

        super(ConvLSTMCell, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_op = conv_op
        self.conv_kwargs = self._default_conv_kwargs.copy()
        if conv_kwargs is not None:
            self.conv_kwargs.update(conv_kwargs)
        self.concat_hidden_and_input = concat_hidden_and_input

        self.add_module("forget", conv_op(in_channels, out_channels, **conv_kwargs))
        self.add_module("input", conv_op(in_channels, out_channels, **conv_kwargs))
        self.add_module("output", conv_op(in_channels, out_channels, **conv_kwargs))
        self.add_module("state", conv_op(in_channels, out_channels, **conv_kwargs))

    def forward(self, input, hidden, cell):

        if self.concat_hidden_and_input:
            input = torch.cat((input, hidden), 1)
        cell = torch.sigmoid(self.forget(input)) * cell + torch.sigmoid(self.input(input)) * torch.tanh(self.state(input))
        hidden = torch.sigmoid(self.output(input)) * torch.tanh(cell)

        return hidden, cell


class TowerRepresentation(ConvModule):

    def __init__(self,
                 in_channels=3,
                 query_channels=7,
                 r_channels=256,
                 conv_op=nn.Conv2d,
                 activation_op=nn.ReLU,
                 activation_kwargs={"inplace": True},
                 pool_op=nn.AdaptiveAvgPool2d,
                 pool_kwargs={"output_size": 1},
                 **kwargs):

        super(TowerRepresentation, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.query_channels = query_channels
        self.r_channels = r_channels
        self.conv_op = conv_op
        self.activation_op = activation_op
        self.activation_kwargs = activation_kwargs if activation_kwargs is not None else {}
        self.pool_op = pool_op
        self.pool_kwargs = pool_kwargs if pool_kwargs is not None else {}

        k = self.r_channels

        self.add_module("conv1", self.conv_op(in_channels, k, kernel_size=2, stride=2))
        self.add_module("conv2", self.conv_op(k, k//2, kernel_size=3, stride=1, padding=1))
        self.add_module("conv2_skip", self.conv_op(k, k, kernel_size=2, stride=2))
        self.add_module("conv3", self.conv_op(k//2, k, kernel_size=2, stride=2))

        self.add_module("conv4", self.conv_op(k + self.query_channels, k//2, kernel_size=3, stride=1, padding=1))
        self.add_module("conv4_skip", self.conv_op(k + self.query_channels, k, kernel_size=3, stride=1, padding=1))
        self.add_module("conv5", self.conv_op(k//2, k, kernel_size=3, stride=1, padding=1))
        self.add_module("conv6", self.conv_op(k, k, kernel_size=1, stride=1))

        self.add_module("activation", self.activation_op(**self.activation_kwargs))
        if self.pool_op is not None:
            self.pool = self.pool_op(**self.pool_kwargs)

    def forward(self, image, query):

        image = self.activation(self.conv1(image))
        skip = self.activation(self.conv2_skip(image))
        image = self.activation(self.conv2(image))
        image = self.activation(self.conv3(image)) + skip

        query = query.view(query.shape[0], -1, 1, 1).repeat(1, 1, *image.shape[2:])

        image = torch.cat((image, query), 1)
        skip = self.activation(self.conv4_skip(image))
        image = self.activation(self.conv4(image))
        image = self.activation(self.conv5(image)) + skip
        image = self.activation(self.conv6(image))

        if self.pool_op is not None:
            image = self.pool(image)

        return image


class GeneratorCore(ConvModule):

    _default_core_kwargs = dict(
        conv_op=nn.Conv2d,
        conv_kwargs=dict(kernel_size=5, stride=1, padding=2),
        concat_hidden_and_input=True
    )

    _default_upsample_kwargs = dict(
        padding=0,
        bias=False
    )

    def __init__(self,
                 query_channels=7,
                 r_channels=256,
                 z_channels=64,
                 h_channels=128,
                 scale=4,
                 core_op=ConvLSTMCell,
                 core_kwargs=None,
                 upsample_op=nn.ConvTranspose2d,
                 upsample_kwargs=None,
                 **kwargs):

        super(GeneratorCore, self).__init__(**kwargs)

        self.query_channels = query_channels
        self.r_channels = r_channels
        self.z_channels = z_channels
        self.h_channels = h_channels
        self.scale = scale
        self.core_op = core_op
        self.core_kwargs = self._default_core_kwargs.copy()
        if core_kwargs is not None:
            self.core_kwargs.update(core_kwargs)
        self.upsample_op = upsample_op
        self.upsample_kwargs = self._default_upsample_kwargs.copy()
        if upsample_kwargs is not None:
            self.upsample_kwargs.update(upsample_kwargs)

        self.add_module("core", core_op(h_channels + query_channels + r_channels + z_channels, h_channels, **self.core_kwargs))
        self.add_module("upsample", upsample_op(h_channels, h_channels, kernel_size=scale, stride=scale, **self.upsample_kwargs))

    def forward(self, viewpoint_query, representation, z, cell, hidden, u):

        batch_size = hidden.shape[0]
        spatial_dims = hidden.shape[2:]

        # we assume all following tensors are either (b,c) or (b,c,1,1,...)
        # we further assume that cell, hidden & u have correct shapes
        if viewpoint_query.shape[2:] != spatial_dims:
            viewpoint_query = viewpoint_query.view(batch_size, -1, *[1, ] * len(spatial_dims)).repeat(1, 1, *spatial_dims)
        if representation.shape[2:] != spatial_dims:
            representation = representation.view(batch_size, -1, *[1, ] * len(spatial_dims)).repeat(1, 1, *spatial_dims)
        if z.shape[2:] != spatial_dims:
            z = z.view(batch_size, -1, *[1, ] * len(spatial_dims)).repeat(1, 1, *spatial_dims)

        cell, hidden = self.core(torch.cat([viewpoint_query, representation, z], 1), cell, hidden)
        u = self.upsample(hidden) + u

        return cell, hidden, u


class InferenceCore(ConvModule):

    _default_core_kwargs = dict(
        conv_op=nn.Conv2d,
        conv_kwargs=dict(
            kernel_size=5,
            stride=1,
            padding=2
        ),
        concat_hidden_and_input=True
    )

    _default_downsample_kwargs = dict(
        padding=0,
        bias=False
    )

    def __init__(self,
                 in_channels=3,
                 query_channels=7,
                 r_channels=256,
                 z_channels=64,
                 h_channels=128,
                 scale=4,
                 core_op=ConvLSTMCell,
                 core_kwargs=None,
                 downsample_op=nn.Conv2d,
                 downsample_kwargs=None,
                 **kwargs):

        super(InferenceCore, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.query_channels = query_channels
        self.r_channels = r_channels
        self.z_channels = z_channels
        self.h_channels = h_channels
        self.scale = scale
        self.core_op = core_op
        self.core_kwargs = self._default_core_kwargs.copy()
        if core_kwargs is not None:
            self.core_kwargs.update(core_kwargs)
        self.downsample_op = downsample_op
        self.downsample_kwargs = self._default_downsample_kwargs.copy()
        if downsample_kwargs is not None:
            self.downsample_kwargs.update(downsample_kwargs)

        self.add_module("core", core_op(in_channels + query_channels + r_channels + 3*h_channels, h_channels, **self.core_kwargs))
        self.add_module("downsample_image", downsample_op(in_channels, in_channels, kernel_size=scale, stride=scale, **self.downsample_kwargs))
        self.add_module("downsample_u", downsample_op(h_channels, h_channels, kernel_size=scale, stride=scale, **self.downsample_kwargs))

    def forward(self, image_query, viewpoint_query, representation, cell, hidden_self, hidden_gen, u):

        batch_size = hidden_self.shape[0]
        spatial_dims = hidden_self.shape[2:]

        # we assume all following tensors are either (b,c) or (b,c,1,1,...)
        # we further assume that image_query, cell, hidden_self, hidden_gen & u have correct shapes
        if viewpoint_query.shape[2:] != spatial_dims:
            viewpoint_query = viewpoint_query.view(batch_size, -1, *[1, ] * len(spatial_dims)).repeat(1, 1, *spatial_dims)
        if representation.shape[2:] != spatial_dims:
            representation = representation.view(batch_size, -1, *[1, ] * len(spatial_dims)).repeat(1, 1, *spatial_dims)

        image_query = self.downsample_image(image_query)
        u = self.downsample_u(u)
        cell, hidden_self = self.core(torch.cat([image_query, viewpoint_query, representation, hidden_gen, u], 1), cell, hidden_self)

        return cell, hidden_self


class GQNDecoder(ConvModule):

    _default_conv_kwargs = dict(
        kernel_size=5,
        stride=1,
        padding=2
    )

    def __init__(self,
                 in_channels=3,
                 query_channels=7,
                 r_channels=256,
                 z_channels=64,
                 h_channels=128,
                 scale=4,
                 core_repeat=12,
                 core_shared=False,
                 generator_op=GeneratorCore,
                 generator_kwargs=None,
                 inference_op=InferenceCore,
                 inference_kwargs=None,
                 conv_op=nn.Conv2d,
                 conv_kwargs=None,
                 output_activation_op=nn.Sigmoid,
                 output_activation_kwargs=None,
                 **kwargs):

        super(GQNDecoder, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.query_channels = query_channels
        self.r_channels = r_channels
        self.z_channels = z_channels
        self.h_channels = h_channels
        self.scale = scale
        self.core_repeat = core_repeat
        self.core_shared = core_shared
        self.generator_op = generator_op
        self.generator_kwargs = {} if generator_kwargs is None else generator_kwargs
        self.inference_op = inference_op
        self.inference_kwargs = {} if inference_kwargs is None else inference_kwargs
        self.conv_op = conv_op
        self.conv_kwargs = self._default_conv_kwargs.copy()
        if conv_kwargs is not None:
            self.conv_kwargs.update(conv_kwargs)
        self.output_activation_op = output_activation_op
        self.output_activation_kwargs = output_activation_kwargs if output_activation_kwargs is not None else {}

        for dict_ in (self.generator_kwargs, self.inference_kwargs):
            dict_.update(dict(
                query_channels=query_channels,
                r_channels=r_channels,
                z_channels=z_channels,
                h_channels=h_channels,
                scale=scale
            ))
        self.inference_kwargs["in_channels"] = in_channels

        if self.core_shared:
            self.add_module("generator_core", self.generator_op(**self.generator_kwargs))
            self.add_module("inference_core", self.inference_op(**self.inference_kwargs))
        else:
            self.add_module("generator_core", nn.ModuleList([self.generator_op(**self.generator_kwargs) for _ in range(self.core_repeat)]))
            self.add_module("inference_core", nn.ModuleList([self.inference_op(**self.inference_kwargs) for _ in range(self.core_repeat)]))

        self.add_module("posterior_net", conv_op(h_channels, 2*z_channels, **self.conv_kwargs))
        self.add_module("prior_net", conv_op(h_channels, 2*z_channels, **self.conv_kwargs))
        self.add_module("observation_net", conv_op(h_channels, in_channels, kernel_size=1, stride=1, padding=0))

        self.add_module("output_activation", output_activation_op(**self.output_activation_kwargs))

    def forward(self, representation, viewpoint_query, image_query):

        batch_size, _, *spatial_dims = image_query.shape
        spatial_dims_scaled = tuple(np.array(spatial_dims) // self.scale)
        kl = 0

        # Increase dimensions
        viewpoint_query = viewpoint_query.view(batch_size, -1, *[1, ] * len(spatial_dims)).repeat(1, 1, *spatial_dims_scaled)
        if representation.shape[2:] != spatial_dims_scaled:
            representation = representation.view(batch_size, -1, *[1, ] * len(spatial_dims)).repeat(1, 1, *spatial_dims_scaled)

        # Reset hidden state
        hidden_g = image_query.new_zeros((batch_size, self.h_channels, *spatial_dims_scaled))
        hidden_i = image_query.new_zeros((batch_size, self.h_channels, *spatial_dims_scaled))

        # Reset cell state
        cell_g = image_query.new_zeros((batch_size, self.h_channels, *spatial_dims_scaled))
        cell_i = image_query.new_zeros((batch_size, self.h_channels, *spatial_dims_scaled))

        # better name for u?
        u = image_query.new_zeros((batch_size, self.h_channels, *spatial_dims))

        for i in range(self.core_repeat):

            if self.core_shared:
                current_generator_core = self.generator_core
                current_inference_core = self.inference_core
            else:
                current_generator_core = self.generator_core[i]
                current_inference_core = self.inference_core[i]

            # Prior
            o = self.prior_net(hidden_g)
            prior_mu, prior_std_pseudo = torch.split(o, self.z_channels, dim=1)
            prior = Normal(prior_mu, F.softplus(prior_std_pseudo))

            # Inference state update
            cell_i, hidden_i = current_inference_core(image_query, viewpoint_query, representation, cell_i, hidden_i, hidden_g, u)

            # Posterior
            o = self.posterior_net(hidden_i)
            posterior_mu, posterior_std_pseudo = torch.split(o, self.z_channels, dim=1)
            posterior = Normal(posterior_mu, F.softplus(posterior_std_pseudo))

            # Posterior sample
            if self.training:
                z = posterior.rsample()
            else:
                z = prior.loc

            # Generator update
            cell_g, hidden_g, u = current_generator_core(viewpoint_query, representation, z, cell_g, hidden_g, u)

            # Calculate KL-divergence
            kl += kl_divergence(posterior, prior)

        image_prediction = self.output_activation(self.observation_net(u))

        return image_prediction, kl

    def sample(self, representation, viewpoint_query, spatial_dims=(64, 64)):

        batch_size = viewpoint_query.shape[0]
        spatial_dims_scaled = tuple(np.array(spatial_dims) // self.scale)

        # Increase dimensions
        viewpoint_query = viewpoint_query.view(batch_size, -1, 1, 1).repeat(1, 1, *spatial_dims_scaled)
        if representation.shape[2:] != spatial_dims_scaled:
            representation = representation.repeat(1, 1, *spatial_dims_scaled)

        # Reset hidden and cell state for generator
        hidden_g = viewpoint_query.new_zeros((batch_size, self.h_channels, *spatial_dims_scaled))
        cell_g = viewpoint_query.new_zeros((batch_size, self.h_channels, *spatial_dims_scaled))

        u = viewpoint_query.new_zeros((batch_size, self.h_channels, *spatial_dims))

        for i in range(self.core_repeat):

            if self.core_shared:
                current_generator_core = self.generator_core
            else:
                current_generator_core = self.generator_core[i]

            o = self.prior_net(hidden_g)
            prior_mu, prior_std_pseudo = torch.split(o, self.z_channels, dim=1)
            prior = Normal(prior_mu, F.softplus(prior_std_pseudo))

            # Prior sample
            z = prior.sample()

            # Update
            cell_g, hidden_g, u = current_generator_core(viewpoint_query, representation, z, cell_g, hidden_g, u)

        return self.output_activation(self.observation_net(u))


class GenerativeQueryNetwork(ConvModule):

    def __init__(self,
                 in_channels=3,
                 query_channels=7,
                 r_channels=256,
                 encoder_op=TowerRepresentation,
                 encoder_kwargs=None,
                 decoder_op=GQNDecoder,
                 decoder_kwargs=None,
                 **kwargs):

        super(GenerativeQueryNetwork, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.query_channels = query_channels
        self.r_channels = r_channels
        self.encoder_op = encoder_op
        self.encoder_kwargs = encoder_kwargs if encoder_kwargs is not None else {}
        self.decoder_op = decoder_op
        self.decoder_kwargs = decoder_kwargs if decoder_kwargs is not None else {}

        self.add_module("encoder", self.encoder_op(self.in_channels, self.query_channels, self.r_channels, **self.encoder_kwargs))
        self.add_module("decoder", self.decoder_op(self.in_channels, self.query_channels, self.r_channels, **self.decoder_kwargs))

    @staticmethod
    def split_batch(images, viewpoints, num_viewpoints):

        # for debugging, if you want to reconstruct known images
        if num_viewpoints == 1:
            return images, viewpoints, images, viewpoints

        # images are (batch_size * num_viewpoints, channels, space)
        # viewpoints are (batch_size * num_viewpoints, channels)
        batch_size = images.shape[0] // num_viewpoints

        # separate input and target
        images = images.view(batch_size, num_viewpoints, *images.shape[1:])
        images, image_query = images[:, :-1], images[:, -1]
        images = images.contiguous().view(batch_size * (num_viewpoints-1), *images.shape[2:])

        viewpoints = viewpoints.view(batch_size, num_viewpoints, *viewpoints.shape[1:])
        viewpoints, viewpoint_query = viewpoints[:, :-1], viewpoints[:, -1]
        viewpoints = viewpoints.contiguous().view(batch_size * (num_viewpoints-1), *viewpoints.shape[2:])

        return images, viewpoints, image_query, viewpoint_query

    def encode(self, images, viewpoints, num_viewpoints):

        representation = self.encoder(images, viewpoints)
        representation = representation.view(-1, max(num_viewpoints, 1), *representation.shape[1:])
        representation = representation.mean(1, keepdim=False)

        return representation

    def forward(self, images, viewpoints, num_viewpoints):
        """Will automatically split input and query. num_viewpoints includes the query viewpoint."""

        images, viewpoints, image_query, viewpoint_query = self.split_batch(images, viewpoints, num_viewpoints)
        representation = self.encode(images, viewpoints, num_viewpoints - 1)
        image_mu, kl = self.decoder(representation, viewpoint_query, image_query)

        return image_mu, image_query, representation, kl

    def sample(self, images, viewpoints, viewpoint_query, num_viewpoints, sigma=None):

        if sigma is None:
            query_sample = viewpoint_query
        else:
            # note that this might produce incorrect viewpoints
            query_sample = Normal(viewpoint_query, sigma).sample()
        representation = self.encode(images, viewpoints, num_viewpoints)
        image_mu = self.decoder.sample(representation, query_sample, images.shape[2:])

        return image_mu
