import gzip
import os
import random
import sys

import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import scipy.ndimage


def load_mnist(path):

    f = gzip.open(path, 'rb')
    if sys.version_info < (3,):
        data = cPickle.load(f)
    else:
        data = cPickle.load(f, encoding='bytes')
    f.close()
    return data


def augmentation(mol, a, aug_th_min, aug_th_max):
    mol = np.array(mol)
    deg_per_rot = 5
    angle = np.random.randint(aug_th_min, aug_th_max, size=(len(mol.shape),))
    for ax in range(angle.size):
        theta = angle[ax] * deg_per_rot * a
        axes = (ax, (ax + 1) % angle.size)
        mol = scipy.ndimage.rotate(
            mol, theta, axes=axes, order=0, reshape=False
        )
    return mol


def make_dirs(path_list):
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)


def make_containing_dirs(path_list):
    for path in path_list:
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


def padding(array, xx, yy, zz=None):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]
    if zz is not None:
        z = array.shape[2]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    if zz is not None:
        c = (zz - z) // 2
        cc = zz - c - z

        return np.pad(
            array, pad_width=((a, aa), (b, bb), (c, cc)), mode='constant'
        )

    else:
        return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


class SaverMNIST:
    def __init__(
        self,
        image_train_path,
        image_test_path,
        csv_train_path,
        csv_test_path,
        image_shape=(32, 32),
        rotation_angle=180,
        data=None,
    ):

        self._image_format = '.npy'

        self.store_image_paths = [image_train_path, image_test_path]
        self.store_csv_paths = [csv_train_path, csv_test_path]
        self.image_shape = image_shape
        self.rotation_angle = rotation_angle

        make_dirs(self.store_image_paths)
        make_containing_dirs(self.store_csv_paths)

        # Load MNIST dataset
        self.data = data

    def run(self):
        for collection, store_image_path, store_csv_path in zip(
            self.data, self.store_image_paths, self.store_csv_paths
        ):
            labels_list = []
            paths_list = []

            for index, (image, label) in enumerate(
                zip(collection[0], collection[1])
            ):
                im = PIL.Image.fromarray(image)
                width, height = im.size
                image_name = str(label) + '_' + str(index) + self._image_format
                image = np.array(image)

                angle = np.random.randint(
                    -self.rotation_angle,
                    +self.rotation_angle,
                    size=(len(image.shape),),
                )
                for ax in range(angle.size):
                    theta = angle[ax]
                    axes = (ax, (ax + 1) % angle.size)
                    image = scipy.ndimage.rotate(
                        image, theta, axes=axes, order=0, reshape=False
                    )

                image = padding(
                    image, self.image_shape[0], self.image_shape[1]
                )

                # Build save path
                save_path = os.path.join(store_image_path, image_name)
                # im.save(save_path, dpi=(300, 300))
                np.save(save_path, image)

                labels_list.append(label)
                paths_list.append(save_path)

            df = pd.DataFrame(
                {'image_paths': paths_list, 'labels': labels_list}
            )

            df.to_csv(store_csv_path)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    # -db DATABSE -u USERNAME -p PASSWORD -size 20
    parser.add_argument("--mnist_file", help="Path to mnist.pkl.gz file")
    parser.add_argument("--output_path", help="Output path to save data")

    args = parser.parse_args()

    mnist_path = args.mnist_file
    output_path = args.output_path

    data_path = os.path.join(output_path, 'mnist_data')

    data = load_mnist(mnist_path)

    mnist_saver = SaverMNIST(
        data=data,
        image_train_path=data_path + '/images_train',
        image_test_path=data_path + '/images_test',
        csv_train_path=data_path + '/train.csv',
        csv_test_path=data_path + '/test.csv',
    )

    # Write files into disk
    mnist_saver.run()

    examples_train = pd.read_csv(data_path + "/train.csv")
    examples_test = pd.read_csv(data_path + "/test.csv")

    array_train_plots = []
    array_test_plots = []

    for i in range(3):
        array_train_plots.append(
            np.load(
                examples_train['image_paths'][
                    random.randint(0, examples_train.shape[0])
                ]
            )
        )
        array_test_plots.append(
            np.load(
                examples_train['image_paths'][
                    random.randint(0, examples_test.shape[0])
                ]
            )
        )

    # create figure
    fig, axis = plt.subplots(2, 3)
    axis[0, 0].imshow(array_train_plots[0])
    axis[0, 1].imshow(array_train_plots[1])
    axis[0, 2].imshow(array_train_plots[2])
    axis[1, 0].imshow(array_test_plots[0])
    axis[1, 1].imshow(array_test_plots[1])
    axis[1, 2].imshow(array_test_plots[2])
    plt.savefig(data_path + "/mnist_examples.png")
