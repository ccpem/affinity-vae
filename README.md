[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/alan-turing-institute/affinity-vae/actions/workflows/tests.yml/badge.svg)](https://github.com/alan-turing-institute/affinity-vae/actions/workflows/tests.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

> Note: This is a development version of the code. The code in the `main`
> branch is a more stable version.

# Affinity-VAE

**Affinity-VAE for disentanglement, clustering and classification of objects in
multidimensional image data**  
Mirecka J, Famili M, Kotanska A, Juraschko N, Costa-Gomes B, Palmer CM,
Thiyagalingam J, Burnley T, Basham M & Lowe AR  
[![doi:10.48550/arXiv.2209.04517](https://img.shields.io/badge/doi-10.48550/arXiv.2209.04517-blue)](https://doi.org/10.48550/arXiv.2209.04517)

## Contents

1. [What Affinity-VAE does](#what-affinity-vae-does)
2. [Installation](#installation)
3. [Preparing your data](#preparing-your-data)
4. [Quick start: a minimal run](#quick-start-a-minimal-run)
5. [Expanded runs: configuration in full](#expanded-runs-configuration-in-full)
6. [All parameters](#all-parameters)
7. [Output directory structure](#output-directory-structure)
8. [Evaluating a trained model](#evaluating-a-trained-model)
9. [Multi-GPU and cluster runs](#multi-gpu-and-cluster-runs)
10. [Inspecting results with Napari](#inspecting-results-with-napari)
11. [Repository layout](#repository-layout)
12. [Development: tests and linting](#development-tests-and-linting)
13. [Known limitations](#known-limitations)

## What Affinity-VAE does

Affinity-VAE is a variational autoencoder for 2D images and 3D volumes (for
example cryo-ET subtomograms). Each input is encoded into:

- a **latent vector** (`latent_dims`) that describes the object's shape or
  identity, and
- an optional **pose vector** (`pose_dims`) that captures pose (such as
  rotation), so the latent vector can be pose-invariant.

The model is trained with a loss made of three parts:

| Term | What it does | Weight |
| --- | --- | --- |
| Reconstruction | MSE (or BCE for binary data) between input and decoded output | 1 |
| KL divergence | Regularises the latent distribution (beta-VAE term) | `beta` |
| Affinity | L1 distance between the cosine similarity of each pair of latent vectors in a batch and the pre-computed similarity of their two classes, read from an **affinity matrix** you provide | `gamma` |

The affinity term organises the latent space so that classes you know to be
similar end up close together. Set `gamma: 0` and `pose_dims: 0` to get a
plain beta-VAE. `beta` and `gamma` can be constant or follow a cyclical
schedule over epochs.

Around the model, the pipeline:

- loads `.mrc`/`.npy` data with the
  [caked](https://github.com/alan-turing-institute/caked) library, takes class
  labels from the file names, splits the data into training and validation sets,
  and can also encode an optional held-out test set,
- trains on CPU, one GPU or several GPUs (through
  [Lightning Fabric](https://lightning.ai/docs/fabric/)),
- regularly trains a classifier (MLP, k-nearest neighbours or logistic regression) on the training
  latents and scores it on the validation latents,
- writes plots (loss, confusion matrices, F1, reconstructions, t-SNE
  embeddings, similarity matrices, interpolations, latent and pose traversals),
  an interactive HTML latent-space explorer, model checkpoints and latent
  metadata,
- tracks the best validation loss and best validation accuracy epochs and saves
  a full set of plots for each,
- evaluates a trained model on new data, including classes it never saw in
  training.

Two decoder families are available: `cnn` (convolutional encoder and decoder)
and `gsd` (convolutional encoder with a differentiable Gaussian-splat decoder).

## Installation

Requirements: Python 3.10 or later, `git` (one dependency, `caked`, is installed
straight from GitHub), and a CUDA GPU if you want GPU training (it is
optional). The current code has been run with Python 3.12, PyTorch 2.13,
Lightning 2.6 and pydantic 2.13.

### pip and a virtual environment (recommended)

```bash
git clone https://github.com/alan-turing-institute/affinity-vae.git
cd affinity-vae

python -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

The `all` extra installs `torch`, `torchvision`, `lightning` and `pydantic>2`.
These are needed to train and are **not** included in a plain `pip install -e .`.

If you are developing, install the test extra and the pre-commit hooks instead:

```bash
python -m pip install -e ".[test]"
python -m pip install pre-commit
pre-commit install
```

Check the install by printing the options:

```bash
python run.py --help
```

### Baskerville (HPC)

```bash
module purge
module load baskerville
module load bask-apps/live
module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0
module load torchvision/0.15.2-foss-2022a-CUDA-11.7.0

python -m venv pyenv_affinity
source pyenv_affinity/bin/activate

git clone https://github.com/alan-turing-institute/affinity-vae.git
cd affinity-vae/
python -m pip install -e ".[baskerville]"
```

See [Multi-GPU and cluster runs](#multi-gpu-and-cluster-runs) for submitting
jobs.

### Napari viewer (separate environment)

The Napari viewer needs `pydantic<2`, which the training code cannot use, so
install it in its own environment with `pip install -e ".[napari]"`. See
[Inspecting results with Napari](#inspecting-results-with-napari).

## Preparing your data

A training run needs three things: a data directory, an affinity matrix and
(optionally) a class list.

### 1. Data directory (`datapath`)

One flat directory of files, all of the same type:

- `datatype: mrc` reads any file whose name contains `.mrc` (so `.mrc` and
  `.mrcs` both work); `datatype: npy` reads `.npy` files.
- Each file holds **one** 2D image or 3D volume. All files must be the same
  size, or use `rescale` to resize them. Each side must be divisible by
  `2^depth` (by 8 with the default `depth: 3`; with `filters`, by 2 to the
  number of layers).
- **The class label is the part of the file name before the first underscore.**
  `1b23_12-321-215.mrcs` has class `1b23`; `2_0_img.npy` has class `2`. The
  rest of the name is kept as metadata.

Optionally, add a `test/` subdirectory with the same layout. Test data is never
used for training. It is encoded every `freq_eval` epochs and shown in the
embedding plots, so test points only appear in plots and checkpoints from
epochs that are a multiple of `freq_eval`. Test classes do not need to appear
in the class list or the affinity matrix.

In plots, test samples are labelled `<class>_test` during training and
`<class>_eval` during evaluation (`--evaluate`), so they are not confused with
training classes of the same name. If test files have no known class, name
them `test_<anything>` (class `test`); they are then labelled plain `test` or
`eval`. Accuracy, confusion matrices and the saved metadata always use the
plain class names.

```
my_data/
├── 1b23_12-321-215.mrcs
├── 1b23_12-78-36.mrcs
├── 1dfo_10-119-219.mrcs
├── ...
├── classes.csv              # optional, can live anywhere
├── affinity.csv             # required for training, can live anywhere
└── test/                    # optional held-out data
    ├── 1fcj_3-10-20.mrcs
    └── ...
```

### 2. Affinity matrix (`affinity`)

A square CSV file. The header row holds the class names, and the rows follow
the same order as the columns. Value `(i, j)` is the similarity between classes
`i` and `j`, in the range -1 to 1, with 1 on the diagonal:

```
a,b,c
1.0,0.14,-0.28
0.14,1.0,0.71
-0.28,0.71,1.0
```

It can contain more classes than you train on. Every training class must be in
it, or the run stops with an error naming the missing classes.

### 3. Class list (`classes`, optional)

A CSV file whose **header row** is the list of classes to train on, e.g.
`1b23,1dfo,1dkg,1e3p`. Files of other classes in `datapath` are ignored (a
good way to hold classes back for evaluation). If you leave this out, every
class found in `datapath` is used.

Example datasets ship with the repository: `tests/testdata_mrc` (3D, 4 training
classes, plus a `test/` folder that includes two unseen classes) and
`tests/testdata_npy` (2D). The [MNIST tutorial](tutorials/README.md) shows how
to build a dataset from scratch.

## Quick start: a minimal run

From the repository root, with the environment active, train for 10 epochs on
the bundled 3D test data:

```bash
python run.py \
  --datapath tests/testdata_mrc \
  --affinity tests/testdata_mrc/affinity_fsc_10.csv \
  --classes tests/testdata_mrc/classes.csv \
  --epochs 10 --batch 8 \
  --out ~/avae_runs --new_out
```

This takes about 20 seconds on one GPU. With no GPU it falls back to the CPU
and logs a warning. On a machine with several GPUs it uses **all** of them by
default; add `--gpu_devices 0` to use just one.

What the options do:

- `--datapath`, `--affinity`, `--classes`: the inputs described above.
- `--epochs 10 --batch 8`: a short run with small batches (the test set is tiny).
- `--out ~/avae_runs`: write all outputs under this directory (it is created if
  missing). Without `--out`, outputs go to the current directory.
- `--new_out`: create a fresh, descriptively named run directory inside `--out`.

Everything else uses its default: a `cnn` model with 7 latent and 3 pose
dimensions, `beta=0.1`, `gamma=2`, an 80/20 train/validation split, and
loss, accuracy, reconstruction and embedding plots every 10 epochs.

While it runs, each epoch logs its losses, for example:

```
Training : Epoch: [10/10] | Loss: ... | Recon: ... | KLdiv: ... | Affin: ... | Beta: 0.100000 | Gamma: 2.000000
Validation : Epoch: [10/10] | Loss: 0.499 | Recon: 0.009 | KLdiv: 0.182 | Affin: 0.236 | ...
------------------->>> Overall accuracy: Train: 1.000000 | Val: 0.666667
```

When it finishes you will find (with `<ts>` the run timestamp, `YYMMDD-HHMM`):

```
~/avae_runs/
└── results_<ts>_model_cnn_lat7_pose3_lr0.001_beta0.1_gamma2.0/
    ├── configs/config_<ts>.yaml        # the full resolved config: re-run with --config_file
    ├── logs/avae_run_log_<ts>.log      # everything printed to the console
    ├── latents/latent_epoch_10.html   # interactive latent-space explorer (open in a browser)
    ├── states/
    │   ├── avae_<ts>_E9_7_3.pt         # model checkpoint (epoch index 9 = 10th epoch)
    │   └── meta_<ts>_E9_L7_P3.pkl      # latents + metadata of every sample, needed for evaluation
    └── plots/
        ├── loss.png, loss_total.png    # loss curves (all terms / total only)
        ├── confusion_trn_norm.png      # normalised confusion matrices (+ .csv, raw and normalised)
        ├── confusion_val_norm.png
        ├── f1.png, f1_trn.csv, f1_val.csv
        ├── recon_trn_in.png / _out.png # input vs reconstruction, central slices
        ├── recon_trn_in.mrc, recon_trn_out.mrc, recon_trn.csv # 3D input and reconstruction grids (same layout) and which sample is where
        ├── recon_val_*                 # same for validation
        ├── embedding_TSNE.png          # t-SNE of the latent space
        ├── embedding_TSNE_pose.png     # t-SNE of the pose space
        ├── best_loss_epoch0010_<ts>/   # per-epoch plots (not loss/F1) from the best-validation-loss epoch
        └── best_acc_epoch0010_<ts>/    # per-epoch plots (not loss/F1) from the best-validation-accuracy epoch
```

The [Output directory structure](#output-directory-structure) section lists
every file a run can produce.

To evaluate the trained model on the `test/` data (including two classes it has
never seen):

```bash
python run.py --evaluate \
  --datapath tests/testdata_mrc \
  --out ~/avae_runs/results_<ts>_model_cnn_lat7_pose3_lr0.001_beta0.1_gamma2.0
```

This writes `*_eval` plots next to the training plots. See
[Evaluating a trained model](#evaluating-a-trained-model).

## Expanded runs: configuration in full

### Three ways to set parameters

Every parameter in [All parameters](#all-parameters) can be set in three places.
Later ones override earlier ones:

1. **Built-in defaults**, from `avae/config.py`.
2. **A YAML config file**, given with `--config_file path.yml`.
3. **The command line**, as `--name value` (or just `--name` for an on/off
   option).

```bash
# all from a config file
python run.py --config_file tests/avae-test-config.yml

# config file, with a few values overridden
python run.py --config_file my_config.yml --epochs 200 --beta 0.5 --new_out
```

When the command line overrides a value from the file, the log shows a
`Command line argument ... is overwriting config value` warning. Unknown keys and
invalid values (for example `model: xyz` or `vis_class: nosuchclass`) stop the run
before training begins, with a message naming the problem.

> **On/off options can only be switched on from the command line.** Writing
> `--vis_rec` sets it to true. To turn off an option that is **on by default**
> (`gpu`, `shift_min`, `no_val_drop`, `vis_best`, `vis_los`, `vis_acc`,
> `vis_rec`, `vis_emb`, `vis_dynamic`), set it to `false` in a config file.
> `filters` (a list) can also only be set in a config file.

Each run saves the fully resolved config (file, command line and defaults
combined, with absolute paths) to `configs/config_<ts>.yaml`. Passing that file
back with `--config_file` reproduces the run.

### An annotated example config

Start from [tests/avae-test-config.yml](tests/avae-test-config.yml) or
[tutorials/mnist_data/mnist_config.yml](tutorials/mnist_data/mnist_config.yml),
or from this one, which sets the commonly changed options:

```yaml
#### Data
datapath: /data/subtomos          # directory of .mrc/.npy files, optional test/ subdir
affinity: /data/affinity.csv      # required for training
classes: /data/classes.csv        # optional: train only on these classes
datatype: mrc                     # mrc or npy
split: 20                         # % of the data used for validation
limit: null                       # cap on the number of files loaded (null = all)

#### Pre-processing
shift_min: true                   # min-max scale each sample to [0, 1]
normalise: false
gaussian_blur: false
rescale: null                     # e.g. 32 resizes every axis to 32 pixels

#### Network
model: cnn                        # cnn or gsd
latent_dims: 8
pose_dims: 1                      # 0 disables the pose channel
depth: 3                          # number of conv layers ...
channels: 64                      # ... and channels in the first one
# filters: [8, 16, 32, 64]        # or give channels per layer explicitly
bnorm_encoder: true
bnorm_decoder: true

#### Training
epochs: 500
batch: 128
learning: 0.001
opt_method: adam
classifier: NN
gpu: true
gpu_devices: "0,1"                # null = all visible GPUs
freq_sta: 10                      # save a checkpoint every N epochs
freq_eval: 10                     # encode test/ every N epochs

#### Loss
beta: 1
gamma: 2
recon_loss: MSE
cyc_method_beta: cycle_sigmoid    # anneal beta from beta_min to beta
beta_min: 0
beta_cycle: 5
cyc_method_gamma: flat            # constant gamma

#### Visualisation
vis_all: true                     # switch on every plot type ...
freq_all: 20                      # ... every 20 epochs (0 = only at best epochs)
vis_format: png
vis_best: true
best_start: 50                    # start tracking best epochs at epoch 50
vis_class: "1b23,1dfo"           # extra latent/pose traversals per listed class
```

Paths can be relative, but they are resolved against the directory you run
from, so absolute paths are safest.

### An example run with every plot

```bash
python run.py \
  --datapath tests/testdata_npy --datatype npy \
  --affinity tests/testdata_npy/affinity_an.csv \
  --classes tests/testdata_npy/classes.csv \
  --epochs 12 --batch 16 --latent_dims 8 --pose_dims 1 \
  --vis_all --freq_all 6 --best_start 10 --vis_epochs \
  --vis_class a,e \
  --out ~/avae_runs --new_out
```

This turns on every plot at epochs 6 and 12, plus at each new best epoch from
epoch 10. It also keeps a copy of each epoch's plots under
`plots/epochs_<ts>/` (`--vis_epochs`) and checkpoints at epochs 6 and 12 (`--freq_all` also sets
`freq_sta` and `freq_eval`).

## All parameters

All of these are available as `--name` on the command line and as `name:` in a
config file. Run `python run.py --help` for the same list with defaults. The
**On** column marks on/off options and their defaults.

### Data

| Parameter | Default | Description |
| --- | --- | --- |
| `datapath` | *required* | Data directory (see [Preparing your data](#preparing-your-data)). If it has a `test/` subdirectory, that is used as the test set. |
| `affinity` | none | Affinity matrix CSV. Required for training, ignored by evaluation. |
| `classes` | none | CSV whose header row lists the classes to train on. Default: all classes in `datapath`. Evaluation uses it to order the classes in its plots. |
| `datatype` | `mrc` | `mrc` or `npy`. |
| `limit` | none | Load at most this many files (taken after class filtering and a random shuffle). Applies to the test set too. |
| `split` | `20` | Percentage of the data held out for validation. |
| `no_val_drop` | on | Keep the last validation batch even if it is smaller than `batch`. |
| `out` | none | Directory for all outputs, created if missing. Default: the current directory. For evaluation, set it to the training run directory. |
| `new_out` | off | Create `results_<ts>_model_<model>_lat<L>_pose<P>_lr<lr>_beta<b>_gamma<g>/` (inside `out` if set) and write everything there. |
| `date_time_run` | now (`YYMMDD-HHMM`) | Run identifier used in all output file names. Set it to give a run a fixed name (e.g. `mnist-tutorial`). Two runs started in the same minute in the same directory share it. |

### Pre-processing

Applied to each sample as it is loaded, in this order: rescale, shift, blur,
normalise.

| Parameter | Default | Description |
| --- | --- | --- |
| `gaussian_blur` | off | Gaussian blur to reduce noise (fixed kernel). |
| `normalise` | off | Normalise each sample. |
| `shift_min` | on | Min-max scale each sample to [0, 1]. Keep this on with `recon_loss: BCE`. |
| `rescale` | none | Resize every axis to this many pixels (e.g. `32`). |

### Network

| Parameter | Default | Description |
| --- | --- | --- |
| `model` | `cnn` | `cnn` (convolutional decoder) or `gsd` (Gaussian-splat decoder). `u`, `a` and `b` are accepted only for evaluating or restarting old checkpoints. |
| `latent_dims` | `7` | Size of the latent (shape) vector. |
| `pose_dims` | `3` | Size of the pose vector. `0` turns pose off. |
| `depth` | `3` | Number of convolutional layers. |
| `channels` | `64` | Channels in the first layer (doubling with each layer). |
| `filters` | none | Channels per layer, e.g. `[8, 16, 32, 64]`. Overrides `depth` and `channels`. **Config file only.** |
| `bnorm_encoder` | off | Batch normalisation in the encoder. |
| `bnorm_decoder` | off | Batch normalisation in the decoder. |
| `n_splats` | `128` | `gsd` only: number of Gaussian splats. |
| `gsd_conv_layers` | `0` | `gsd` only: number of output channels of the convolution layers added after the splat decoder (`0` = no convolution layers). |

### Training

| Parameter | Default | Description |
| --- | --- | --- |
| `epochs` | `100` | Number of epochs. |
| `batch` | `16` | Batch size (per GPU when using several GPUs). |
| `learning` | `0.001` | Learning rate. |
| `opt_method` | `adam` | `adam`, `sgd` or `asgd`. |
| `classifier` | `NN` | Classifier fitted on the training latents to measure accuracy: `NN` (scikit-learn MLP, adam solver), `KNN` (k-nearest neighbours) or `LR` (logistic regression). All are tuned by grid search on macro F1. |
| `gpu` | on | Use CUDA GPUs if present, otherwise CPU. |
| `gpu_devices` | none | Comma-separated GPU indices, e.g. `0,1,3`. Default: all visible, working GPUs. |
| `strategy` | `auto` | Lightning multi-GPU strategy: `auto`, `ddp`, `deepspeed` or `fsdp`. Falls back to `auto` on a single GPU. |
| `freq_sta` | `10` | Save a checkpoint and metadata file to `states/` every N epochs. |
| `freq_eval` | `10` | Encode the `test/` set every N epochs (if there is one). |

### Loss

| Parameter | Default | Description |
| --- | --- | --- |
| `beta` | `0.1` | Weight of the KL term (the maximum when annealing). |
| `gamma` | `2` | Weight of the affinity term (the maximum when annealing). |
| `recon_loss` | `MSE` | `MSE`, or `BCE` for data in [0, 1]. |
| `klreduction` | `mean` | `mean` or `sum` reduction of the KL term. |

### beta and gamma schedules

The same options exist for `gamma`: replace `beta` with `gamma` in each name.

| Parameter | Default | Description |
| --- | --- | --- |
| `cyc_method_beta` | `flat` | `flat` (constant at `beta`), `cycle_linear`, `cycle_sigmoid`, `cycle_cosine` (repeat a rise from `beta_min` to `beta` `beta_cycle` times), or `ramp` (a single rise lasting `epochs / beta_cycle × beta_ratio` epochs, then constant). |
| `beta_min` | `0` | Starting value of each cycle. |
| `beta_cycle` | `4` | Number of cycles over the run. |
| `beta_ratio` | `0.5` | Fraction of each cycle spent rising; the rest stays at `beta`. |
| `beta_load` | none | `.npy` file of per-epoch values (length must equal `epochs`). Overrides all the above. [tools/create_scheduling_func.ipynb](tools/create_scheduling_func.ipynb) helps build one. |

Use `--vis_cyc` to plot the resulting schedules.

### Restart and evaluation

| Parameter | Default | Description |
| --- | --- | --- |
| `evaluate` | off | Evaluate a trained model instead of training (see [below](#evaluating-a-trained-model)). |
| `restart` | off | Resume training from a checkpoint. Currently broken, see [Known limitations](#known-limitations). |
| `state` | none | Checkpoint (`.pt`) to load. Default: the newest one in `states/`. |
| `meta` | none | Metadata file (`.pkl`) to load for evaluation. Default: the newest one in `states/`. |

### Visualisation: which plots

Plots marked *per-epoch* are drawn every `freq_<name>` epochs (counting from
1), at the best epochs when `vis_best` is on, and at no other time. With
`freq_<name>: 0` a plot is only drawn at the best epochs (see
[Visualisation: frequencies](#visualisation-frequencies)).

| Parameter | Default | When | Output (in `plots/` unless noted) |
| --- | --- | --- | --- |
| `vis_los` | on | every epoch from epoch 2 | `loss`, `loss_total` |
| `vis_acc` | on | per-epoch (`freq_acc`) | `confusion_{trn,val}_norm` + CSVs, `f1` + `f1_{trn,val}.csv` |
| `vis_rec` | on | per-epoch (`freq_rec`) | `recon_{trn,val}_in/_out`; for 3D also `recon_{trn,val}_in.mrc` / `_out.mrc` (same grid layout) + `.csv` |
| `vis_emb` | on | per-epoch (`freq_emb`) | `embedding_TSNE`, `embedding_TSNE_pose` |
| `vis_dynamic` | on | per-epoch (`freq_dynamic`) | `latents/latent_epoch_<N>.html` (interactive; large for big datasets) |
| `vis_sim` | off | per-epoch (`freq_sim`) | `similarity_mean_{trn,val}`: mean cosine similarity between classes in latent space |
| `vis_int` | off | per-epoch (`freq_int`) | `interpolations_trn`: 10x10 latent grid between real encodings of 4 random classes; with only 2 or 3 classes, a row of 10 steps between 2 classes (skipped with 1 class). Pose fixed to the mean of the corners' poses |
| `vis_dis` | off | per-epoch (`freq_dis`) | `disentanglement-latent_avg_trn`: traversal of each latent dimension, from -2 to +2 standard deviations in 15 steps (+ `disentanglement-latent_<class>_trn` with `vis_class`) |
| `vis_pos` | off | per-epoch (`freq_pos`) | `disentanglement-pose_avg_trn`: traversal of each pose dimension, same range (+ `disentanglement-pose_<class>_trn` with `vis_class`) |
| `vis_his` | off | once | `classes_distribution_{train,validation,evaluation}` |
| `vis_aff` | off | once | `affinity_matrix` (training classes highlighted) |
| `vis_cyc` | off | once | `hyperaparam_beta`, `hyperaparam_gamma` |
| `vis_all` | none | — | Set to on/off to switch every option above at once. Options you give on the command line take precedence. |

For 3D data, the interpolation, traversal and pose plots are saved as `.mrc`
grids (open them in Napari or ChimeraX), plus a PNG (`vis_format`) of the central
slice of every volume in the same grid, with `PNG_EDGE_CROP` (3) pixels cropped
from each in-plane edge. The `.mrc` is not cropped.

### Visualisation: settings

| Parameter | Default | Description |
| --- | --- | --- |
| `vis_format` | `png` | `png`, `pdf` or `svg`. Vector formats stay sharp and are fastest for large matrices. |
| `vis_print` | off | Render matrix plots at 300 dpi (print quality) instead of screen resolution. |
| `vis_best` | on | Whenever the validation loss or validation accuracy reaches a new best (from `best_start` on), draw every enabled per-epoch plot regardless of `freq_*`, into `plots/best_loss_epochXXXX_<ts>/` or `plots/best_acc_epochXXXX_<ts>/`. Only the latest folder of each kind per run is kept. Validation accuracy is computed every epoch from `best_start` on, which costs some time per epoch. |
| `best_start` | `10` | First epoch considered by `vis_best`. A warning is logged if it is later than `epochs`. |
| `vis_epochs` | off | Also keep every epoch's plots in `plots/epochs_<ts>/epoch_XXXX/` (`plots/` still holds the latest). |
| `vis_class` | none | Comma-separated classes, e.g. `1b23,1dfo`. With `vis_dis`, one latent traversal figure per class; with `vis_pos`, one pose traversal figure per class (from the class mean latent and the class pose mean and spread). A class not in the data stops the run when the plot is first made. |

The classes in `vis_class` are checked when the first per-class plot is made; a class that is not in the data stops the run with an error listing the available classes. A class with a single sample or no spread is still plotted, with a warning.

### Visualisation: frequencies

`freq_acc`, `freq_rec`, `freq_emb`, `freq_dynamic`, `freq_sim`, `freq_int`,
`freq_dis` and `freq_pos` (all default `10`) set how often, in epochs, each
per-epoch plot is drawn. `freq_all` sets all of them, **and `freq_sta` and
`freq_eval`**, to one value. As with `vis_all`, values given explicitly on the
command line take precedence.

A frequency of `0` turns the periodic plot off. With `vis_best` on, that plot
is then drawn only when the validation loss or accuracy reaches a new best
(from `best_start`). To get only best-epoch plots:

```yaml
vis_best: true
best_start: 10
freq_all: 0        # per-epoch plots only at best epochs
```

`freq_all: 0` does not touch `freq_sta` or `freq_eval`, so checkpoints and test
encoding keep their own schedule (test points only appear in a best-epoch
embedding if `freq_eval` lands on that epoch). The loss plot (`vis_los`) has no
frequency and is redrawn every epoch. With `vis_best` off, a plot with
frequency `0` is never drawn and a warning is logged.

### Other

| Parameter | Default | Description |
| --- | --- | --- |
| `config_file` | none | YAML config file (command line only). |
| `debug` | off | Verbose (DEBUG) logging. |

## Output directory structure

All outputs go to the run directory: `--out`, then the `results_...` folder if
`--new_out` is set, otherwise the current directory. A full training run
produces:

```
<run dir>/
├── configs/
│   └── config_<ts>.yaml                 # resolved config of each run (one file per <ts>)
├── logs/
│   └── avae_run_log_<ts>.log            # full log (appended if <ts> repeats)
├── states/                              # every freq_sta epochs
│   ├── avae_<ts>_E<e>_<L>_<P>.pt        # checkpoint: weights, model object, optimiser, loss
│   │                                    #   history, best epochs. <e> is 0-based: E9 = epoch 10
│   └── meta_<ts>_E<e>_L<L>_P<P>.pkl     # DataFrame: one row per train/val/test sample with
│                                        #   file name, class, mode, latent + pose vectors, images
├── latents/
│   └── latent_epoch_<N>.html           # interactive Altair latent explorer (hover for images)
└── plots/
    ├── loss.<fmt>, loss_total.<fmt>     # updated every epoch
    ├── f1.<fmt>, f1_trn.csv, f1_val.csv
    ├── affinity_matrix.<fmt>            # vis_aff (+ affinity_matrix_classes.txt for caching)
    ├── classes_distribution_*.<fmt>     # vis_his
    ├── hyperaparam_{beta,gamma}.<fmt>   # vis_cyc
    ├── <latest per-epoch plots>         # confusion_*, recon_*, embedding_*, similarity_*,
    │                                    #   interpolations_*, disentanglement-*
    ├── best_loss_epochXXXX_<ts>/        # vis_best: per-epoch plots at the best val-loss epoch
    ├── best_acc_epochXXXX_<ts>/         # vis_best: per-epoch plots at the best val-accuracy epoch
    └── epochs_<ts>/epoch_XXXX/          # vis_epochs: per-epoch plots of every plotted epoch
```

Notes:

- `<ts>` is `date_time_run`; `<L>`/`<P>` are `latent_dims`/`pose_dims`;
  `<fmt>` is `vis_format`; `XXXX` is the 1-based epoch, zero-padded.
- Per-epoch plots are overwritten in `plots/` each time they are drawn, so
  `plots/` always holds the latest set. Use `vis_epochs` to keep every epoch's plots.
- Loss, F1 and the once-per-run plots are only written to `plots/`, never into
  the `best_*` or `epochs_*` folders.
- Running several trainings in the same directory is safe: each run's `best_*`
  and `epochs_*` folders carry its own `<ts>` and are not removed by other runs.
- Evaluation adds `*_eval` files to the same directory (see below).
- `latents/*.html` files embed a thumbnail of every sample and can become very
  large; consider `vis_dynamic: false` for big training runs and switch it on
  for evaluation.

## Evaluating a trained model

Evaluation encodes new data with a trained model, compares it with the
training latent space and writes evaluation plots. Run it **in the training run
directory**, either with `--out <run dir>` or by changing into it:

```bash
python run.py --evaluate \
  --datapath /data/new_subtomos \
  --out ~/avae_runs/results_<ts>_...
```

To reuse every training setting (data type, pre-processing, `vis_*`), pass the
saved config and override what changes:

```bash
cd ~/avae_runs/results_<ts>_...
python /path/to/affinity-vae/run.py --config_file configs/config_<ts>.yaml \
  --evaluate --datapath /data/new_subtomos
```

The saved config keeps the training `date_time_run`, `out` and `new_out`, so
this evaluation overwrites `configs/config_<ts>.yaml` and appends to the
training log. When it uses `new_out`, it finds the run directory by rebuilding
its name, so do not override any parameter that is part of that name
(`date_time_run`, `model`, `latent_dims`, `pose_dims`, `learning`, `beta`,
`gamma`). Otherwise it opens a new, empty directory with no `states/`.

How it works:

- **Data**: everything in `datapath`. If `datapath` has a `test/`
  subdirectory, **that subdirectory is used instead**. Labels come from file
  names as usual. Classes unseen in training are allowed and reported
  separately.
- **Model**: `--state`, or by default the newest `states/*.pt` (files with
  `eval` in the name are skipped). The number of pose dimensions is read from
  the checkpoint file name, so keep the original names.
- **Training latents**: `--meta`, or by default the newest `states/*.pkl`.
- **Pre-processing** options (`shift_min`, `normalise`, ...) must match
  training. The saved config takes care of this.
- **GPU**: evaluation runs on one device: the first index in `gpu_devices`, or GPU 0.

Outputs (in the same run directory):

```
plots/confusion_trn_eval_norm.<fmt>, confusion_eval_norm.<fmt> (+ .csv)
plots/f1_eval.<fmt>, f1_trn_eval.csv, f1_eval.csv
plots/embedding_TSNE_eval.<fmt>                    # evaluation data only
plots/embedding_TSNE_trn_eval_comparison.<fmt>   # training + evaluation together
plots/recon_eval_in/_out.<fmt> (+ recon_eval_in.mrc/_out.mrc/.csv for 3D)
plots/*_eval*                                      # dis/pos/int/sim plots if enabled
latents/latent_epoch_1_eval.html
states/meta_<ts>_E<e>_L<L>_P<P>_eval.pkl           # training metadata + evaluation rows
```

The log reports accuracy on the training set, on seen classes only, and on all
classes including unseen ones:

```
Overall accuracy: Train: 1.000000 | Val : 1.000000 | Val with unseen labels: 0.300000
```

## Multi-GPU and cluster runs

With `gpu: true` and no `gpu_devices`, all visible GPUs are checked and every
working one is used, in a data-parallel run through Lightning Fabric
(`strategy: auto` normally selects DDP). `batch` is per GPU, so the training set
must hold at least `batch × number of GPUs` samples. Plots and checkpoints are
written by rank 0 only. More than 4 devices are spread over nodes assuming 4
GPUs per node (as on Baskerville).

On a SLURM cluster, edit and submit [tools/slurm_run.sh](tools/slurm_run.sh):

```bash
sbatch tools/slurm_run.sh
```

It loads the Baskerville modules and runs
`srun python run.py --config_file <config> --new_out`. Change the account,
QoS, environment path and config path at the top of the script first.

## Inspecting results with Napari

[scripts/run_napari_model_view.py](scripts/run_napari_model_view.py) opens an
interactive Napari viewer for a trained model: decode any point of the latent
embedding, and move along individual latent and pose dimensions to see the
decoder's reconstruction.

### Why a separate environment

Napari needs `pydantic<2`, while the training code needs `pydantic>2`. They
cannot share one environment, so install the viewer in its own virtual
environment. Do not use it for training, and do not install `[napari]` into
your training environment. From the repository root:

```bash
python -m venv env_napari
source env_napari/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[napari]"
```

The `napari` extra also installs `umap-learn`, which is not a dependency of the
training environment.

### Running it

```bash
python scripts/run_napari_model_view.py \
  --model_file <run>/states/<model>.pt \
  --meta_file  <run>/states/<meta>.pkl
```

- `--model_file`: a checkpoint from the run's `states/` directory (the best or
  the latest one).
- `--meta_file`: the metadata `.pkl` from the same run (also in `states/`). It
  holds each sample's `id`, latents (`lat*`), poses (`pos*`) and stored 2D
  embedding (`emb*`).
- `--manifold umap|load` (default `umap`):
  - `umap` fits a UMAP on the stored latents when the viewer starts. A click
    anywhere on the map is decoded through UMAP's `inverse_transform`, so you can
    explore regions with no data. It is slower, and the map differs from the
    t-SNE plots made in training. t-SNE cannot be inverted, so it is not offered.
  - `load` uses the embedding stored in the meta file and snaps a click to the
    nearest stored sample, whose latent is decoded. It is fast and needs no
    `umap-learn`, but only reproduces latents that exist in the data.
- `--latent_dims`, `--pose_dims`: optional integers that override the values
  read from the model (`encoder.mu` / `encoder.pose_fc`). Set them if the model
  has no such attributes.

### The widgets

- **Latents tab**: one slider per latent dimension, ranging over the min and max
  of that dimension in the meta file. Moving one re-decodes the volume.
- **Pose group**, depending on `pose_dims`:
  - `4`: axis-angle control. A `theta` slider (0 to 2 pi) and an `axes`
    choice (X, Y or Z) for the rotation axis.
  - `1` (or any other value): one slider per pose dimension (`pose0`, ...),
    ranging over the values of that pose column in the meta file. With the
    splat decoder, a single pose value is a rotation about z.
  - `0`: no pose controls.
- **Manifold tab**: a density map of the embedding with class labels at the
  class centroids. Click to decode that point; the latent sliders jump to the
  decoded latent.

Here is the interface on a model trained on the Shrec protein dataset:
![NapariUMAP](images/Napari1.png) ![NapariLatents](images/Napari2.png)

### Common errors

- `ImportError: ... umap-learn`: you are in an environment without the `napari`
  extra. Run `pip install -e ".[napari]"`, or use `--manifold load`.
- `pydantic` or `napari` import errors, or training failing in this
  environment: you mixed the environments. Use the Napari one only for the viewer.
- `Model does not have pose/latent attributes`: pass `--pose_dims` and
  `--latent_dims`.
- Blank or wrong-shaped reconstruction: `--pose_dims` / `--latent_dims` do not
  match the trained model.

## Repository layout

```
affinity-vae/
├── run.py                   # command-line entry point (train or evaluate)
├── avae/                    # the package
│   ├── config.py            # AffinityConfig: every parameter, default and check
│   ├── train.py             # training loop, best-epoch tracking, checkpoints
│   ├── evaluate.py          # evaluation of a trained model
│   ├── data.py              # data loading (via caked), affinity matrix, labels
│   ├── models.py            # model selection (cnn / gsd)
│   ├── base.py              # model base classes
│   ├── encoders/, decoders/ # network building blocks
│   ├── loss.py              # reconstruction + KL + affinity loss
│   ├── cyc_annealing.py     # beta / gamma schedules
│   ├── utils_learning.py    # classifiers, accuracy, t-SNE, metadata tables
│   ├── utils_gpu.py         # CPU / GPU / multi-GPU setup
│   ├── vis.py               # all plots
│   ├── utils_vis.py         # plot folders, saving, grids
│   └── napari.py            # Napari viewer widgets
├── configs/                 # example config for tools/create_subtomo.py
├── tests/                   # pytest suite, example config, test datasets
│   ├── avae-test-config.yml
│   ├── testdata_mrc/        # small 3D dataset (+ test/ with unseen classes)
│   └── testdata_npy/        # small 2D dataset
├── tutorials/               # MNIST tutorial (README, data builder, config)
├── scripts/                 # Napari viewer, subtomogram creation
├── tools/                   # SLURM script, data prep, schedule and plot notebooks
└── images/                  # README images
```

`tools/create_subtomo.py` (run through `scripts/run_create_subtomo.py` with
`configs/avae-create_subtomo_test_config.yml`) cuts, filters and augments
subtomograms from full tomograms.
[tools/plotting_post_process.ipynb](tools/plotting_post_process.ipynb) helps
analyse run outputs.

## Development: tests and linting

From the repository root with the `[test]` extra installed:

```bash
python -m pytest -q tests                               # full suite (~2.5 min)
python -m pytest -q tests/test_vis.py -k <name>         # a subset
pre-commit run --files <changed files>                  # black, isort, mypy, file checks
```

Tests marked `integration` run short end-to-end trainings. Pytest is set up in
`pyproject.toml` to turn warnings into errors, apart from a list of known ones.

## Known limitations

- **`--restart` is broken**: resuming training stops with a `NameError` (the
  `state` variable is not set in `avae/train.py`). Until this is fixed, start a
  new run instead.
- On/off options that default to on, and `filters`, can only be changed in a
  config file (see
  [Three ways to set parameters](#three-ways-to-set-parameters)).
- Class labels must be the file-name prefix before the first `_`.
- Training needs an affinity matrix. Without one, the run stops with an error.
