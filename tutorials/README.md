# AffinityVAE Tutorial

## Running AffinityVAE on the MNIST dataset

### 1. Install AffinityVAE
To start, clone the AffinityVAE repository and switch to the `affinity-vae` directory
As described in the [README](../README.md), AffinityVAE can be installed using
pip:

```bash
python -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ."[all]"
```

### 2. Download the MNIST dataset

The MNIST dataset can be downloaded from:
https://figshare.com/articles/dataset/mnist_pkl_gz/13303457

In this tutorial, we assume that the dataset is downloaded to this `tutorials`
directory. Make sure that the downloaded file is indeed `mnist.pkl.gz` and that
it hasn't been decompressed by your browser (this could happen in Safari).

### Create the MNSIT dataset for affinityVAE

We need an MNIST dataset with augmentation and rotations, and saved with a
structure that affinityVAE can read (a subdirectory for training data, and
another for testing). This can be done by running the following command:

```bash
 python mnist_saver.py --mnist_file mnist.pkl.gz --output_path .
```

here the first argument is the path to the downloaded MNIST dataset, and the
second argument is the path to the directory where the processed dataset will be
saved. In this example, we are saving the dataset in the current directory under
[mnist_data](mnist_data). There you can also find a visualization of a few
random samples of the dataset in the `mnist_examples.png` file for validation.

Affinity-VAE is a highly configurable model (to see the full list of
configurables run `python absolute/path/to/affinity-vae/run.py --help`). In this
tutorial we will use a yaml file [mnist_config.yml](mnist_data/mnist_config.yml)
to configure our run.

Now we are ready to run AffinityVAE on the MNIST dataset. For this we need the
data divided into training and test sets, we need a configuration file
(`mnist_config.yml`), and we need the files `classes_mnist.csv` (list of labels
to use in training) and `affinity_mnist.csv` (affinity matrix for the classes
defined above), which are provided in his tutorial. All these files can be found
in the [mnist_data](mnist_data) directory.

First thing you need to do is to modify the `mnist_config.yml` file to point to
the absolute paths of the data, classes, and affinity files. Currently, the
paths in the config file are relative to this directory but this can cause issues
when running the code from a different directory. To modify the paths, open the
`mnist_config.yml` file and change the following lines:

```yaml
datapath: /absolute/path/to/mnist_data/images_train/
classes: /absolute/path/to/mnist_data/classes_mnist.csv
affinity: /absolute/path/to/mnist_data/affinity_mnist.csv
```

In general, we recommend to work with absolute paths to avoid issues.

To train AffinityVAE on our rotated MNIST dataset, run the following command:

```bash
python /absolute/path/to/affinity-vae/run.py --config_file /absolute/path/to/mnist_data/mnist_config.yml --new_out
```

This will train AffinityVAE on the MNIST dataset and save the results in a new
timestamped directory created by the `--new_out` flag. In this case the run is
configured by the `mnist_config.yml` file.

You can also configure the run by passing arguments to the `run.py` script as
shown in the main README file and in the following example:

```bash
python path/to/affinity-vae/run.py --config_file /absolute/path/to/mnist_data/mnist_config.yml --beta 0.1 --gamma 0.01 --learning 0.001 --epochs 200 --new_out
```

Here the command line arguments override the values in the config file.

The config file provided here has optimal parameters for the MNIST dataset, so
we recommend to use it as it is.

Once the training finishes you can evaluate the model on unseen data using the
test set by stepping into the new directory and running the following. The outputs from the test run will be saved in the existing plots directory created during the training run, and will have the suffix ```evl_``` in the filename.

```bash
cd path/to/new_out
python path/to/affinity-vae/run.py --config_file /absolute/path/to/mnist_data/mnist_config.yml --datapath /absolute/path/to/mnist_data/images_test/ --evaluate
```

_Note_: During training we've left the class `9` out, so we can use it for
evaluation and see where it fits in the affinity organised latent space.

You can also restart training from a checkpoint by running

```bash
cd path/to/new_out
python path/to/affinity-vae/run.py --config_file /absolute/path/to/mnist_data/mnist_config.yml --restart --epochs 2000 --datapath /absolute/path/to/mnist_data/images_train/
```

here epochs are set to 2000 to continue training for 1000 extra epochs (assuming
initial training ran for the 1000 epochs defined in the config file).

# Outputs of the training and evaluation runs

The training run will create a directory with the following structure:

```
new_out
├── configs # copy of the config file used for the run for reproducibility
├── logs # run logs
├── plots # plots and data of the training and evaluation metrics
├── latents # html files latent space of the training and test sets, these files can be very large, so we recommend enabling them only at evaluation time (using the --vis_dynamic flag)
├── states #saving checkpoints of the models and the training latent space to be use for evaluation or restart training


```
