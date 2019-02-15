from collections import namedtuple


DatasetInfo = namedtuple('DatasetInfo', ['image_size', 'seq_length', 'train_instances', 'test_instances'])

all_datasets = dict(
    jaco=DatasetInfo(image_size=64, seq_length=11, train_instances=7200000, test_instances=800000),
    mazes=DatasetInfo(image_size=84, seq_length=300, train_instances=108000, test_instances=12000),
    rooms_free_camera_with_object_rotations=DatasetInfo(image_size=128, seq_length=10, train_instances=10170000, test_instances=630000),
    rooms_ring_camera=DatasetInfo(image_size=64, seq_length=10, train_instances=10800000, test_instances=1200000),
    rooms_free_camera_no_object_rotations=DatasetInfo(image_size=64, seq_length=10, train_instances=10800000, test_instances=1200000),
    shepard_metzler_5_parts=DatasetInfo(image_size=64, seq_length=15, train_instances=810000, test_instances=200000),
    shepard_metzler_7_parts=DatasetInfo(image_size=64, seq_length=15, train_instances=810000, test_instances=200000)
)
