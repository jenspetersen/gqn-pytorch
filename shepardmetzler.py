#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

data_dir = "../../shepard_metzler"
file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(file_dir, data_dir)


def split(N=5, seed=1):

    indices = np.arange(810000)
    r = np.random.RandomState(seed)
    r.shuffle(indices)
    num = len(indices) // N
    splits = []
    for i in range(N):
        if i < (N-1):
            splits.append(sorted(indices[i*num:(i+1)*num]))
        else:
            splits.append(sorted(indices[i*num:]))
    return splits


def load(mode="train"):

    shape = [810000, 15, 64, 64, 3]
    if mode == "test":
        shape[0] = 200000
    shape = tuple(shape)

    images = np.memmap(os.path.join(data_dir, "{}_images.npy".format(mode)), mode="r", dtype=np.uint8, shape=shape)
    viewpoints = np.load(os.path.join(data_dir, "{}_viewpoints.npy".format(mode)))

    return {"data": images, "viewpoints": viewpoints}


def transform_viewpoint(v):
    """
    Transforms the viewpoint vector into a consistent
    representation
    """

    return np.concatenate([v[:, :3],
                           np.cos(v[:, 3:4]),
                           np.sin(v[:, 3:4]),
                           np.cos(v[:, 4:5]),
                           np.sin(v[:, 4:5])], 1)


class LinearBatchGenerator(SlimDataLoaderBase):

    def __init__(self,
                 data,
                 batch_size,
                 dtype=np.float32,
                 num_viewpoints=15,  # both input and query
                 shuffle_viewpoints=False,
                 data_order=None,
                 **kwargs):

        super(LinearBatchGenerator, self).__init__(data, batch_size, **kwargs)
        self.dtype = dtype
        self.num_viewpoints = num_viewpoints
        self.shuffle_viewpoints = shuffle_viewpoints

        self.current_position = 0
        self.was_initialized = False
        if self.number_of_threads_in_multithreaded is None:
            self.number_of_threads_in_multithreaded = 1
        if data_order is None:
            self.data_order = np.arange(data["viewpoints"].shape[0])
        else:
            self.data_order = data_order

        self.num_restarted = 0

    def reset(self):

        self.current_position = self.thread_id * self.batch_size
        self.was_initialized = True
        self.rs = np.random.RandomState(self.num_restarted)
        self.num_restarted = self.num_restarted + 1

    def __len__(self):

        return len(self.data_order)

    def generate_train_batch(self):

        if not self.was_initialized:
            self.reset()
        if self.current_position >= len(self):
            self.reset()
            raise StopIteration
        batch = self.make_batch(self.current_position)
        self.current_position += self.number_of_threads_in_multithreaded * self.batch_size
        return batch

    def make_batch(self, idx):

        batch_images = []
        batch_viewpoints = []
        data_indices = []
        viewpoint_indices = []

        if self.num_viewpoints == "random":
            num_viewpoints = self.rs.randint(2, 16)
        else:
            num_viewpoints = self.num_viewpoints

        while len(batch_images) < self.batch_size:

            idx = idx % len(self.data_order)
            idx_data = self.data_order[idx]

            viewpoint_indices_current = np.arange(15)
            # for linear generator we leave the existing viewpoint order
            if self.shuffle_viewpoints:
                self.rs.shuffle(viewpoint_indices_current)
            viewpoint_indices_current = viewpoint_indices_current[:num_viewpoints]
            viewpoint_indices.append(viewpoint_indices_current)

            batch_images.append(np.array(self._data["data"][idx_data, viewpoint_indices_current]))
            batch_viewpoints.append(self._data["viewpoints"][idx_data, viewpoint_indices_current])
            data_indices.append(idx_data)

            idx += 1

        batch_images = np.stack(batch_images)\
            .astype(self.dtype)\
            .reshape(self.batch_size * num_viewpoints, 64, 64, 3)\
            .transpose(0, 3, 1, 2)
        batch_viewpoints = np.stack(batch_viewpoints)\
            .astype(np.float32)\
            .reshape(self.batch_size * num_viewpoints, -1)
        batch_viewpoints = transform_viewpoint(batch_viewpoints)
        data_indices = np.array(data_indices)
        viewpoint_indices = np.array(viewpoint_indices)

        # images are saved as uint8, so we need to normalize
        if self.dtype != np.uint8:
            batch_images /= 255.

        return {"data": batch_images,
                "viewpoints": batch_viewpoints,
                "num_viewpoints": num_viewpoints,
                "data_indices": data_indices,
                "viewpoint_indices": viewpoint_indices}


class RandomOrderBatchGenerator(LinearBatchGenerator):

    def __init__(self,
                 *args,
                 num_viewpoints="random",
                 shuffle_viewpoints=True,
                 infinite=True,
                 **kwargs):

        super(RandomOrderBatchGenerator, self).__init__(*args, num_viewpoints=num_viewpoints, **kwargs)
        self.infinite = infinite

    def reset(self):

        super(RandomOrderBatchGenerator, self).reset()
        self.rs.shuffle(self.data_order)

    def generate_train_batch(self):

        if not self.was_initialized:
            self.reset()
        if self.current_position >= len(self):
            self.reset()
            if not self.infinite:
                raise StopIteration
        batch = self.make_batch(self.current_position)
        self.current_position += self.number_of_threads_in_multithreaded * self.batch_size
        return batch


class RandomBatchGenerator(RandomOrderBatchGenerator):
    pass
