import os
import scipy.io as sio
import random
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def get_list_imgs(data_path):
    list_imgs = os.listdir(data_path)
    list_imgs = [os.path.join(data_path, img) for img in list_imgs]
    random.shuffle(list_imgs)
    return list_imgs


def to_float32(x): return tf.cast(x, dtype=tf.float32)
def double_input(x): return (x, x)


class Dataset(tf.data.Dataset):

    def _generator(data_path):

        list_imgs = get_list_imgs(data_path)

        for img_path in list_imgs:
            x = sio.loadmat(img_path)['img']
            yield x

    def __new__(cls, input_size=(512, 512, 31), data_path="../data/kaist/train"):
        output_signature = tf.TensorSpec(shape=input_size, dtype=tf.float32)

        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=output_signature,
            args=(data_path,)
        )


def prepare(ds, shuffle=False, augment=False, batch_size=32, cache=False):

    ds = ds.map(lambda x: to_float32(x), num_parallel_calls=AUTOTUNE)

    if cache:
        ds = ds.cache('')

    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(batch_size)

    ds = ds.map(lambda x: double_input(x), num_parallel_calls=AUTOTUNE)

    return ds.prefetch(buffer_size=AUTOTUNE)


def load_dataset(data_path='./data/kaist_dataset', train_params=None, test_params=None):

    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    
    train_ds = Dataset(data_path=train_path)
    test_ds  = Dataset(data_path=test_path)
    
    train_ds = prepare(train_ds, **train_params)
    test_ds = prepare(test_ds, **test_params)

    return train_ds, test_ds
