import numpy as np
import time
import torch
from torch import nn, optim, distributions
import torch.backends.cudnn as cudnn
cudnn.benchmark = False

from batchgenerators.dataloading import MultiThreadedAugmenter
from trixi.util import Config, ResultLogDict
from trixi.experiment import PytorchExperiment

from model import GenerativeQueryNetwork
from util import get_default_experiment_parser, set_seeds, run_experiment
from loader import shepardmetzler5


DESCRIPTION = """This experiment just tries to reproduce GQN results,
                 specifically for the Shepard-Metzler-5 dataset."""


def make_defaults():

    DEFAULTS = Config(

        # Base
        name="gqn",
        description=DESCRIPTION,
        n_epochs=1000000,
        batch_size=36,
        batch_size_val=36,
        seed=1,
        device="cuda",

        # Data
        split_val=3,  # index for set of 5
        split_test=4,  # index for set of 5
        data_module=shepardmetzler5,
        data_dir=None,  # will be set for data_module if not None
        debug=0,  # 1 for single repeating batch, 2 for single viewpoint (i.e. reconstruct known images)
        generator_train=shepardmetzler5.RandomBatchGenerator,
        generator_val=shepardmetzler5.LinearBatchGenerator,
        num_viewpoints_val=8,  # use this many viewpoints in validation
        shuffle_viewpoints_val=False,
        augmenter=MultiThreadedAugmenter,
        augmenter_kwargs={"num_processes": 8},

        # Model
        model=GenerativeQueryNetwork,
        model_kwargs={
            "in_channels": 3,
            "query_channels": 7,
            "r_channels": 256,
            "encoder_kwargs": {
                "activation_op": nn.ReLU
            },
            "decoder_kwargs": {
                "z_channels": 64,
                "h_channels": 128,
                "scale": 4,
                "core_repeat": 12
            }
        },
        model_init_weights_args=None,  # e.g. [nn.init.kaiming_normal_, 1e-2],
        model_init_bias_args=None,  # e.g. [nn.init.constant_, 0],

        # Learning
        optimizer=optim.Adam,
        optimizer_kwargs={"weight_decay": 1e-5},
        lr_initial=5e-4,
        lr_final=5e-5,
        lr_cutoff=16e4,  # lr is increased linearly in cutoff epochs
        sigma_initial=2.0,
        sigma_final=0.7,
        sigma_cutoff=2e4,  # sigma is increased linearly in cutoff epochs
        kl_weight_initial=0.05,
        kl_weight_final=1.0,
        kl_weight_cutoff=1e5,  # kl_weight is increased linearly in cutoff epochs
        nll_weight=1.0,

        # Logging
        backup_every=10000,
        validate_every=1000,
        validate_subset=0.01,  # validate only this percentage randomly
        show_every=100,
        val_example_samples=10,  # draw this many random samples for last validation item
        test_on_val=True,  # test on the validation set

    )

    SHAREDCORES = Config(
        model_kwargs={"decoder_kwargs": {"core_shared": True}}
    )

    MODS = {
        "SHAREDCORES": SHAREDCORES
    }

    return {"DEFAULTS": DEFAULTS}, MODS


class GQNExperiment(PytorchExperiment):

    def setup(self):

        set_seeds(self.config.seed, "cuda" in self.config.device)

        self.setup_data()
        self.setup_model()

        self.config.epoch_str_template = "{:0" + str(len(str(self.config.n_epochs))) + "d}"
        self.clog.show_text(self.model.__repr__(), "Model")

    def setup_data(self):

        c = self.config

        if c.data_dir is not None:
            c.data_module.data_dir = c.data_dir

        # set actual data
        self.data_train_val = c.data_module.load("train")
        self.data_test = c.data_module.load("test")

        # train, val, test split
        indices_split = c.data_module.split()
        indices_val = indices_split[c.split_val]
        indices_test = indices_split[c.split_test]
        indices_train = []
        for i in range(5):
            if i not in (c.split_val, c.split_test):
                indices_train += indices_split[i]
        indices_train = sorted(indices_train)

        # for debugging we only use a single batch and validate on training data
        if c.debug > 0:
            indices_train = indices_train[:c.batch_size]
            indices_val = indices_train
            indices_test = indices_test[:c.batch_size_val]

        # construct generators
        self.generator_train = c.generator_train(
            self.data_train_val,
            c.batch_size,
            data_order=indices_train,
            num_viewpoints=1 if c.debug == 2 else "random",
            shuffle_viewpoints=not c.debug,
            number_of_threads_in_multithreaded=c.augmenter_kwargs.num_processes)
        self.generator_val = c.generator_val(
            self.data_train_val,
            c.batch_size_val,
            data_order=indices_val,
            num_viewpoints=1 if c.debug == 2 else c.num_viewpoints_val,
            shuffle_viewpoints=c.shuffle_viewpoints_val,
            number_of_threads_in_multithreaded=c.augmenter_kwargs.num_processes)
        self.generator_test = c.generator_val(
            self.data_test,
            c.batch_size_val,
            data_order=indices_test,
            num_viewpoints=1 if c.debug == 2 else c.num_viewpoints_val,
            number_of_threads_in_multithreaded=c.augmenter_kwargs.num_processes)

        # construct augmenters (no actual augmentation at the moment, just multithreading)
        self.augmenter_train = c.augmenter(self.generator_train, None, **c.augmenter_kwargs)
        self.augmenter_val = c.augmenter(self.generator_val, None, **c.augmenter_kwargs)
        self.augmenter_test = c.augmenter(self.generator_test, None, **c.augmenter_kwargs)

    def setup_model(self):

        c = self.config

        # intialize model and weights
        self.model = c.model(**c.model_kwargs)
        if c.model_init_weights_args is not None and hasattr(self.model, "init_weights"):
            self.model.init_weights(*c.model_init_weights_args)
        if c.model_init_bias_args is not None and hasattr(self.model, "init_bias"):
            import IPython
            IPython.embed()
            self.model.init_bias(*c.model_init_bias_args)

        # optimization
        self.optimizer = c.optimizer(self.model.parameters(), lr=c.lr_initial, **c.optimizer_kwargs)
        self.lr = c.lr_initial
        self.sigma = c.sigma_initial

    def _setup_internal(self):

        super(GQNExperiment, self)._setup_internal()
        self.elog.save_config(self.config, "config")  # default PytorchExperiment only saves self._config_raw

        # we want a results dictionary with running mean, so close default and construct new
        self.results.close()
        self.results = ResultLogDict("results-log.json", base_dir=self.elog.result_dir, mode="w", running_mean_length=self.config.show_every)

    def prepare(self):

        # move everything to selected device
        for name, model in self.get_pytorch_modules().items():
            model.to(self.config.device)

    def train(self, epoch):

        c = self.config

        t0 = time.time()

        # set learning rates, sigmas, loss weights
        self.train_prepare(epoch)

        # get data
        data = next(self.augmenter_train)
        data["data"] = torch.from_numpy(data["data"]).to(dtype=torch.float32, device=c.device)
        data["viewpoints"] = torch.from_numpy(data["viewpoints"]).to(dtype=torch.float32, device=c.device)

        # forward
        image_pred, image_query, representation, kl = self.model(data["data"], data["viewpoints"], data["num_viewpoints"])
        loss_elbo, loss_nll, loss_kl = self.criterion(image_pred, image_query, kl)

        # backward
        loss_elbo.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        training_time = time.time() - t0

        # use data dictionary as training summary
        data["data"] = data["data"].cpu()
        data["viewpoints"] = data["viewpoints"].cpu()
        data["image_query"] = image_query.cpu()  # also in "data" but we're lazy
        data["image_pred"] = image_pred.cpu()
        data["loss_elbo"] = loss_elbo.item()
        data["loss_nll"] = loss_nll.item()
        data["loss_kl"] = loss_kl.item()
        data["training_time"] = training_time

        self.train_log(data, epoch)

    def train_prepare(self, epoch):

        c = self.config

        # sets parameters as is done in the paper, additionally start with lower KL weight
        self.lr = max(c.lr_final + (c.lr_initial - c.lr_final) * (1 - epoch / c.lr_cutoff), c.lr_final)
        self.sigma = max(c.sigma_final + (c.sigma_initial - c.sigma_final) * (1 - epoch / c.sigma_cutoff), c.sigma_final)
        _lr = self.lr * np.sqrt(1 - 0.999**(epoch+1)) / (1 - 0.9**(epoch+1))
        for group in self.optimizer.param_groups:
            group["lr"] = _lr
        self.nll_weight = c.nll_weight
        self.kl_weight = min(c.kl_weight_final, c.kl_weight_initial + (c.kl_weight_final - c.kl_weight_initial) * epoch / c.kl_weight_cutoff)

        self.model.train()
        self.optimizer.zero_grad()

    def criterion(self, image_predicted, image_query, kl, batch_mean=True):

        # mean over batch but sum over individual
        nll = -distributions.Normal(image_predicted, self.sigma).log_prob(image_query)
        # nll = nn.MSELoss(reduction="none")(image_predicted, image_query)
        nll = nll.view(nll.shape[0], -1)
        kl = kl.view(kl.shape[0], -1)
        if batch_mean:
            nll = nll.mean(0)
            kl = kl.mean(0)
        nll = nll.sum(-1)
        kl = kl.sum(-1)

        elbo = self.nll_weight * nll + self.kl_weight * kl

        return elbo, nll, kl

    def train_log(self, summary, epoch):

        _backup = (epoch + 1) % self.config.backup_every == 0
        _show = (epoch + 1) % self.config.show_every == 0

        self.elog.show_text("{}/{}: {}".format(epoch, self.config.n_epochs, summary["training_time"]), name="Training Time")

        # add_result will show graphs and log to json file at the same time
        self.add_result(summary["loss_elbo"], "loss_elbo", epoch, "Loss", plot_result=_show, plot_running_mean=True)
        self.add_result(summary["loss_nll"], "loss_nll", epoch, "Loss", plot_result=_show, plot_running_mean=True)
        self.add_result(summary["loss_kl"], "loss_kl", epoch, "Loss", plot_result=_show, plot_running_mean=True)

        self.make_images(summary["image_query"],
                         "reference",
                         epoch,
                         save=_backup,
                         show=_show)
        self.make_images(summary["image_pred"],
                         "reconstruction",
                         epoch,
                         save=_backup,
                         show=_show)

    def validate(self, epoch):

        c = self.config

        if (epoch+1) % c.validate_every == 0:

            with torch.no_grad():

                t0 = time.time()
                self.model.eval()

                validation_scores = []
                info = {}  # holds info on score array axes
                info["dims"] = ["Object Index", "Loss"]
                info["coords"] = {"Object Index": [], "Loss": ["NLL", "KL", "ELBO"]}

                example_output_shown = False
                for d, data in enumerate(self.augmenter_val):

                    # this ensures we always validate at least one item even for very small subset ratios
                    if c.validate_subset not in (False, None, 1.) and c.debug == 0:
                        rand_number = np.random.rand()
                        if rand_number < 1 - c.validate_subset:
                            if not (d * c.batch_size_val >= len(self.generator_val) - 1 and len(validation_scores) == 0):
                                continue

                    # get data
                    data["data"] = torch.from_numpy(data["data"]).to(dtype=torch.float32, device=c.device)
                    data["viewpoints"] = torch.from_numpy(data["viewpoints"]).to(dtype=torch.float32, device=c.device)

                    # forward
                    image_pred, image_query, representation, kl = self.model(data["data"], data["viewpoints"], data["num_viewpoints"])
                    loss_elbo, loss_nll, loss_kl = self.criterion(image_pred, image_query, kl, batch_mean=False)

                    # use data dict as summary dict
                    data["data"] = data["data"].cpu()
                    data["viewpoints"] = data["viewpoints"].cpu()
                    data["image_query"] = image_query.cpu()
                    data["image_pred"] = image_pred.cpu()
                    data["loss_elbo"] = loss_elbo.cpu()
                    data["loss_nll"] = loss_nll.cpu()
                    data["loss_kl"] = loss_kl.cpu()

                    current_scores = np.array([data["loss_nll"].cpu().numpy(),
                                               data["loss_kl"].cpu().numpy(),
                                               data["loss_elbo"].cpu().numpy()]).T
                    validation_scores.append(current_scores)
                    info["coords"]["Object Index"].append(data["data_indices"])

                    self.make_images(data["image_query"],
                                     "val/{}_reference".format(d),
                                     epoch,
                                     save=True,
                                     show=False)
                    self.make_images(data["image_pred"],
                                     "val/{}_prediction".format(d),
                                     epoch,
                                     save=True,
                                     show=False)
                    self.make_images(data["data"],
                                     "val/{}_seen".format(d),
                                     epoch,
                                     save=True,
                                     show=False,
                                     images_per_row=data["data"].shape[0] // c.batch_size_val)

                    # only show one validation item
                    if not example_output_shown:
                        self.make_images(data["image_query"],
                                         "val_reference",
                                         epoch,
                                         save=False,
                                         show=True)
                        self.make_images(data["image_pred"],
                                         "val_prediction",
                                         epoch,
                                         save=False,
                                         show=True)
                        self.make_images(data["data"],
                                         "val_seen",
                                         epoch,
                                         save=False,
                                         show=True,
                                         images_per_row=data["data"].shape[0] // c.batch_size_val)
                        example_output_shown = True

                validation_time = time.time() - t0
                validation_scores = np.concatenate(validation_scores, 0)
                info["coords"]["Object Index"] = np.concatenate(info["coords"]["Object Index"], 0)

                # there can be duplicates in the last batch
                for i in range(c.batch_size_val):
                    if info["coords"]["Object Index"][-(i+1)] not in info["coords"]["Object Index"][:-(i+1)]:
                        break
                if i > 0:
                    validation_scores = validation_scores[:-i]
                    info["coords"]["Object Index"] = info["coords"]["Object Index"][:-i]

                summary = {}
                summary["validation_time"] = validation_time
                summary["validation_scores"] = validation_scores
                summary["validation_info"] = info

                self.validate_log(summary, epoch)

                # draw a few different samples for the last data item
                # item could have been skipped, so we might need to transfer again
                if c.val_example_samples > 0:
                    if isinstance(data["data"], np.ndarray):
                        data["data"] = torch.from_numpy(data["data"]).to(dtype=torch.float32, device=c.device)
                        data["viewpoints"] = torch.from_numpy(data["viewpoints"]).to(dtype=torch.float32, device=c.device)
                    images_context, viewpoints_context, _, viewpoint_query =\
                        self.model.split_batch(data["data"],
                                               data["viewpoints"],
                                               data["num_viewpoints"])
                    images_context = images_context.to(device=c.device)
                    viewpoints_context = viewpoints_context.to(device=c.device)
                    viewpoint_query = viewpoint_query.to(device=c.device)

                    samples = []
                    for i in range(c.val_example_samples):
                        samples.append(self.model.sample(images_context, viewpoints_context, viewpoint_query, data["num_viewpoints"] - 1, self.sigma).cpu())
                    samples = torch.cat(samples, 0)
                    # samples should now be (batch * samples, 3, 64, 64)
                    self.make_images(samples, "samples", epoch, save=True, show=True, images_per_row=c.batch_size_val)

    def validate_log(self, summary, epoch):

        epoch_str = self.config.epoch_str_template.format(epoch)
        validation_scores_mean = np.nanmean(summary["validation_scores"], 0)

        self.elog.save_numpy_data(summary["validation_scores"], "validation/{}.npy".format(epoch_str))
        self.elog.save_dict(summary["validation_info"], "validation/{}.json".format(epoch_str))
        self.elog.show_text("{}/{}: {}".format(epoch, self.config.n_epochs, summary["validation_time"]), name="Validation Time")

        self.add_result(float(validation_scores_mean[2]), "loss_elbo_val", epoch, "Loss")
        self.add_result(float(validation_scores_mean[0]), "loss_nll_val", epoch, "Loss")
        self.add_result(float(validation_scores_mean[1]), "loss_kl_val", epoch, "Loss")

    def _end_epoch_internal(self, epoch):

        self.save_results()
        if (epoch+1) % self.config.backup_every == 0:
            self.save_temp_checkpoint()

    def make_images(self,
                    images,
                    name,
                    epoch,
                    save=False,
                    show=True,
                    images_per_row=None):

        n_images = images.shape[0]
        if images_per_row is None:
            images_per_row = int(np.sqrt(n_images))

        if show and self.vlog is not None:
            self.vlog.show_image_grid(images, name,
                                      image_args={"normalize": True,
                                                  "nrow": images_per_row,
                                                  "pad_value": 1})
        if save and self.elog is not None:
            name = self.config.epoch_str_template.format(epoch) + "/" + name
            self.elog.show_image_grid(images, name,
                                      image_args={"normalize": True,
                                                  "nrow": images_per_row,
                                                  "pad_value": 1})

    def test(self):

        pass


if __name__ == '__main__':

    parser = get_default_experiment_parser()
    args, _ = parser.parse_known_args()
    DEFAULTS, MODS = make_defaults()
    run_experiment(GQNExperiment,
                   DEFAULTS,
                   args,
                   mods=MODS,
                   explogger_kwargs=dict(folder_format="{experiment_name}_%Y%m%d-%H%M%S"),
                   globs=globals(),
                   resume_save_types=("model", "simple", "th_vars", "results"))
