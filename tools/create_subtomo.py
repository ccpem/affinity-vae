import os
import warnings

import mrcfile
import numpy as np
import pandas as pd
import scipy.ndimage


def create_subtomo(
    input_path,
    output_path,
    datatype,
    annot_path,
    vox_size=[32, 32, 32],
    bandpass=False,
    low_freq=0,
    high_freq=15,
    gaussian_blur=False,
    normalise=False,
    add_noise=False,
    noise_int=0,
    padding=None,
    augment=None,
    aug_th_min=-45,
    aug_th_max=45,
):
    """Function to train an AffinityVAE model. The inputs are training configuration parameters. In this function the
    data is loaded, selected and split into training, validation and test sets, the model is initialised and trained
    over epochs, the results are evaluated visualised and saved and the epoch level with a frequency configured with
    input parameters.

    Parameters
    ----------
    input_path: str
        Path to the folder containing the full tomogram/image.
    output_path: str
        Path to the folder for output subtomograms
    datatype : str
        data file formats : mrc, npy
    annot_path : str
        path to the folder containing the name of the particles and their x,y,z coordinates
        file names should be identical to that of the full tomograms with the extension of txt
        for mat of the txt files is ,  'class','z','y','x','cx','cy','cz
    vox_size: list
        size of each subtomogram voxel given as a list where vox_size: [x,y,x]
    bandpass: bool
        Apply band pass filter to the full tomogram before extracting the subtomograms
    low_freq: float
        Lower frequency threshold for the band pass filter
    high_freq: float
        higher frequency threshold for the band pass filter
    gaussian_blur: bool
        if True, Gaussian bluring is applied to the input before being passed to the model.
        This is added as a way to remove noise from the input data.
    normalise: bool
        In True, the input data is normalised before being passed to the model.
    add_noise: bool
        Add noise to images, this can be used for benchmarking
    noise_int: int
        noise intensity
    padding: list
        size of padding boxes
    augment: int
        perform the given number of aumentation on each voxel before saving
    aug_th_max: int
        The minimum value of the augmentation range in degrees
    aug_th_min: int
        The maximum value of the augmentation range in degrees
    """

    a = 0
    # the name of all full tomograms
    file_list = [f for f in os.listdir(input_path) if "." + datatype in f]
    print("List of tomograms provided:", file_list)
    # the name of all annotations for tomograms
    annot_list = [f for f in os.listdir(input_path) if "." + "txt" in f]
    print("List of annotation files provided:", annot_list)

    if len(annot_list) != len(file_list):
        raise RuntimeError(
            "number of full tomograms do not match the number of annotation files"
        )

    for t, tomo in enumerate(file_list):
        print("#####################################")
        print("#####################################")
        print(f"{t}: segmenting tomogram {tomo}")
        print("#####################################")
        print("#####################################")

        data = read_mrc(os.path.join(input_path, tomo))
        if bandpass:
            print("creating band pass filter")
            bandpass_filter_image = bandpass_filter(
                data.shape, low_freq, high_freq
            )
            print("Applying band pass filter")
            data = apply_bandpass_filter(data, bandpass_filter_image)

        if add_noise:
            data = add_g_noise(data, noise_int)

        if gaussian_blur:
            print("gaussian blur is not implemented yet")
            data = data

        if normalise:
            data = normalisiation(data)

        particle_df = particles_GT(
            os.path.join(annot_path, f"{tomo[:-4]}.txt"), output_path
        )

        particle_df = delete_detection_in_edges(
            particle_df, data.shape, vox_size
        )

        for index, protein in particle_df.iterrows():
            name = protein["class"]
            s = [int(n / 2) for n in vox_size]
            mol = data[
                protein["x"] - s[0] : protein["x"] + s[0],
                protein["y"] - s[1] : protein["y"] + s[1],
                protein["z"] - s[2] : protein["z"] + s[2],
            ]
            for a in range(augment):
                mol = augmentation(mol, a, aug_th_min, aug_th_max)
                if padding is not None:
                    mol = padding_mol(mol, padding)

                mol_file_name = (
                    f"{name}_{tomo[:-4]}_{index}_Th{a}_n{noise_int}.mrc"
                )
                print("saving:", mol_file_name)
                mrcfile.write(
                    os.path.join(output_path, mol_file_name),
                    mol,
                    overwrite=True,
                )


def normalisiation(x):
    x = (x - x.min()) / (x.max() - x.min())
    return x


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


def bandpass_filter(image_size, bp_low, bp_high):

    bandpass_filter = np.zeros((image_size), dtype=np.float32)
    if len(image_size) == 2:
        for u in range(image_size[0]):
            for v in range(image_size[1]):
                D = np.sqrt(
                    (u - image_size[0] / 2) ** 2 + (v - image_size[1] / 2) ** 2
                )
                if D <= bp_low:
                    bandpass_filter[u, v] = 1
                elif D >= bp_high:
                    bandpass_filter[u, v] = 1
    elif len(image_size) == 3:
        for u in range(image_size[0]):
            for v in range(image_size[1]):
                for w in range(image_size[2]):
                    D = np.sqrt(
                        (u - image_size[0] / 2) ** 2
                        + (v - image_size[1] / 2) ** 2
                        + (w - image_size[2] / 2) ** 2
                    )

                    if D <= bp_low:
                        bandpass_filter[u, v, w] = 1
                    elif D >= bp_high:
                        bandpass_filter[u, v, w] = 1
    return bandpass_filter


def apply_bandpass_filter(image, bandpass_filter):
    F = np.fft.fftn(image)
    Fshift = np.fft.fftshift(F)
    Gshift = Fshift * bandpass_filter
    G = np.fft.ifftshift(Gshift)
    filtered_image = abs(np.fft.ifftn(G))
    return filtered_image.astype("float32")


def add_g_noise(input_a, scale):
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var**0.5

    gaussian = np.random.normal(
        mean, sigma, (input_a.shape[0], input_a.shape[1], input_a.shape[1])
    )

    output_a = input_a + gaussian * scale
    return output_a.astype(np.float32)


def read_mrc(path):
    """
    Takes a path and read a mrc file convert the data to np array
    """
    warnings.simplefilter(
        "ignore"
    )  # to mute some warnings produced when opening the tomos
    with mrcfile.open(path, mode="r+", permissive=True) as mrc:
        mrc.update_header_from_data()

        mrc.header.map = mrcfile.constants.MAP_ID
        mrc = mrc.data

    with mrcfile.open(path) as mrc:
        data = np.array(mrc.data)
    return data


def particles_GT(annot, output_path):

    labels = np.loadtxt(annot, dtype="str")

    labels = np.reshape(labels, (-1, 7))
    df = pd.DataFrame(labels)
    df.columns = ["class", "z", "y", "x", "cx", "cy", "cz"]

    df = df.astype({"x": "int"})
    df = df.astype({"y": "int"})
    df = df.astype({"z": "int"})
    proteins = df["class"].unique()

    df.drop(df[df["class"] == "vesicle"].index, inplace=True)

    with open(f"{output_path}/classes.csv", "w") as f:
        f.write(",".join(proteins))
    return df


def delete_detection_in_edges(df, data_shape, vox_size):
    # drop the particles that are appearing in the edges
    df.drop(df[df["x"] > data_shape[0] - vox_size[0]].index, inplace=True)
    df.drop(df[df["y"] > data_shape[1] - vox_size[1]].index, inplace=True)
    df.drop(df[df["z"] > data_shape[2] - vox_size[2]].index, inplace=True)
    df.drop(df[df["x"] < vox_size[0]].index, inplace=True)
    df.drop(df[df["y"] < vox_size[1]].index, inplace=True)
    df.drop(df[df["z"] < vox_size[2]].index, inplace=True)
    return df


def padding_mol(array, padding_size):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]
    z = array.shape[2]

    xx = padding_size[0]
    yy = padding_size[1]
    zz = padding_size[2]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    c = (zz - z) // 2
    cc = zz - c - z

    return np.pad(
        array, pad_width=((a, aa), (b, bb), (c, cc)), mode="constant"
    )
