# Adapted from https://github.com/l3robot/gqn_datasets_translator

import os
import numpy as np
from numpy.lib.format import open_memmap
import tensorflow as tf
import subprocess as sp
from .datasets import all_datasets
import argparse as ap

tf.logging.set_verbosity(tf.logging.ERROR)  # disable annoying logging
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable gpu

_pose_dim = 5


def collect_files(path, ext=None, key=None):
    if key is None:
        files = sorted(os.listdir(path))
    else:
        files = sorted(os.listdir(path), key=key)

    if ext is not None:
        files = [f for f in files if os.path.splitext(f)[-1] == ext]

    return [os.path.join(path, fname) for fname in files]


def convert_record(record, info):
    print(record)

    path, filename = os.path.split(record)
    images, viewpoints = process_record(record, info)

    return images, viewpoints


def process_record(record, info):
    engine = tf.python_io.tf_record_iterator(record)

    images = []
    viewpoints = []
    for i, data in enumerate(engine):
        image, viewpoint = convert_to_numpy(data, info)
        images.append(image)
        viewpoints.append(viewpoint)

    return np.stack(images), np.stack(viewpoints)


def process_images(example, seq_length, image_size):
    """Instantiates the ops used to preprocess the frames data."""
    images = tf.concat(example['frames'], axis=0)
    images = tf.map_fn(tf.image.decode_jpeg, tf.reshape(images, [-1]),
                       dtype=tf.uint8, back_prop=False)
    shape = (image_size, image_size, 3)
    images = tf.reshape(images, (-1, seq_length) + shape)
    return images


def process_poses(example, seq_length):
    """Instantiates the ops used to preprocess the cameras data."""
    poses = example['cameras']
    poses = tf.reshape(poses, (-1, seq_length, _pose_dim))
    return poses


def convert_to_numpy(raw_data, info):
    seq_length = info.seq_length
    image_size = info.image_size

    feature = {'frames': tf.FixedLenFeature(shape=seq_length, dtype=tf.string),
               'cameras': tf.FixedLenFeature(shape=seq_length * _pose_dim, dtype=tf.float32)}
    example = tf.parse_single_example(raw_data, feature)

    images = process_images(example, seq_length, image_size)
    poses = process_poses(example, seq_length)

    return images.numpy().squeeze(), poses.numpy().squeeze()


if __name__ == '__main__':

    tf.enable_eager_execution()

    parser = ap.ArgumentParser(description='Convert gqn tfrecords to gzipped numpy arrays.')
    parser.add_argument('base_dir', nargs=1,
                        help='base directory of gqn dataset')
    parser.add_argument('dataset', nargs=1,
                        help='datasets to convert, eg. shepard_metzler_5_parts')
    parser.add_argument('-n', '--first-n', type=int, default=None,
                        help='convert only the first n tfrecords if given')
    parser.add_argument('-m', '--mode', type=str, default='train',
                        help='whether to convert train or test')
    parser.add_argument("-o", "--output_dir", type=str, default=os.getcwd(), help="Output directory, default current working dir")
    parser.add_argument("-c", "--compression_cores", type=int, default=8, help="Use this many cores for compression.")
    args = parser.parse_args()

    base_dir = os.path.expanduser(args.base_dir[0])
    dataset = args.dataset[0]
    output_dir = os.path.join(args.output_dir, dataset)
    os.makedirs(output_dir, exist_ok=True)

    print(f'base_dir: {base_dir}')
    print(f'dataset: {dataset}')
    print(f'output_dir: {output_dir}')

    info = all_datasets[dataset]
    data_dir = os.path.join(base_dir, dataset)
    records = collect_files(os.path.join(data_dir, args.mode), '.tfrecord')

    if args.first_n is not None:
        records = records[:args.first_n]

    images_shape = (getattr(info, "{}_instances".format(args.mode)), info.seq_length, info.image_size, info.image_size, 3)
    viewpoints_shape = (getattr(info, "{}_instances".format(args.mode)), info.seq_length, _pose_dim)

    images_arr = open_memmap(os.path.join(output_dir, "{}_images.npy".format(args.mode)), mode="w+", dtype=np.uint8, shape=images_shape)
    viewpoints_arr = open_memmap(os.path.join(output_dir, "{}_viewpoints.npy".format(args.mode)), mode="w+", dtype=np.float32, shape=viewpoints_shape)

    print(f'converting {len(records)} records in {dataset}/{args.mode}')

    index = 0
    for r, record in enumerate(records):
        images, viewpoints = convert_record(record, info)
        images_arr[index:index+images.shape[0]] = images
        viewpoints_arr[index:index+images.shape[0]] = viewpoints
        index += images.shape[0]

    del images_arr
    del viewpoints_arr

    sp.check_output(["pigz", "--fast", "-p", "{}".format(args.compression_cores), os.path.join(output_dir, "{}_viewpoints.npy".format(args.mode))])
    sp.check_output(["pigz", "--fast", "-p", "{}".format(args.compression_cores), os.path.join(output_dir, "{}_images.npy".format(args.mode))])

    print('Done')
