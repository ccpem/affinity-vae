[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/alan-turing-institute/affinity-vae/actions/workflows/tests.yml/badge.svg)](https://github.com/alan-turing-institute/affinity-vae/actions/workflows/tests.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

> Note: This a development version of the code. The code in the `main` branch is
> a more stable version of the code.

# Affinity-VAE

**Affinity-VAE for disentanglement, clustering and classification of objects in
multidimensional image data**  
Mirecka J, Famili M, Kotanska A, Juraschko N, Costa-Gomes B, Palmer CM,
Thiyagalingam J, Burnley T, Basham M & Lowe AR  
[![doi:10.48550/arXiv.2209.04517](https://img.shields.io/badge/doi-10.48550/arXiv.2209.04517-blue)](https://doi.org/10.48550/arXiv.2209.04517)

## Installation

### Installing with pip + virtual environments

> Note: This has been tested in the `develop` branch.

You can install the libraries needed for this package on a
[fresh virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
with the following:

```
python -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ."[all]"
```

If you are developing code, you should be able to run pre-commits and tests,
therefore the best installation option after setting up the virtual environment
is:

```
python -m pip install -e ."[test]"
```

run tests locally from this same working directory as installation (root of this
repository):

```
python -m pytest -s -W ignore
```

> Note: This is the preferred option for running on macOS laptops.

> Warning: M1 macOS can not do
> [pytorch paralelisation](https://github.com/pytorch/pytorch/issues/70344). A
> temporary solution for this is to modify the code on the DataLoaders in
> data.py to `num_workers=0` in order to run the code. Otherwise, you will get
> the error:
> `AttributeError: Can't pickle local object 'ProteinDataset.__init__.<locals>.<lambda>'`.

### Installing with pyenv in Baskerville

The following is the recommended way of installing all libraries in Baskervile.

```

module purge
module load baskerville
module load bask-apps/live

module load PyTorch/2.0.1-foss-2022a-CUDA-11.7.0
module load torchvision/0.15.2-foss-2022a-CUDA-11.7.0

python -m venv pyenv_affinity
source pyenv_affinity/bin/activate

git clone https://github.com/alan-turing-institute/affinity-vae.git
cd  affinity-vae/
python -m pip install -e ."[baskerville]"
```

you can send a SLURM job to the cluster using the script in [tools](tools/slurm_run.sh), with the following command:

```bash
sbatch slurm_run.sh
```


## Running Affinity-VAE: A quick start

We have a [tutorial](tutorials/README.md) on how to run Affinity-VAE on the
MNIST dataset. We recommend to start there for the first time you run
Affinity-VAE.

<details>
<summary><i>Affinity-VAE configuration parameters</i></summary>

Affinity-VAE has a running script (`run.py`) that allows you to configure and
run the code. You can look at the available configuration options by running:

```
python run.py --help
```

which will give you:

```
Usage: run.py [OPTIONS]

Options:
  --config_file PATH
  -d, --datapath TEXT             Path to training data.
  -dtype, --datatype TEXT         Type of the data: mrc, npy
  -dbg, --debug                   Run in debug mode.
  -res, --restart                 Is the calculation restarting from a
                                  checkpoint.
  -st, --state TEXT               The saved model state to be loaded for
                                  evaluation/resume training.
  -mt, --meta TEXT                The saved meta file to be loaded for
                                  regenerating dynamic plots.
  -lm, --limit INTEGER            Limit the number of samples loaded (default
                                  None).
  -sp, --split INTEGER            Train/val split in %.
  -newo, --new_out                Create new output directory where to save
                                  the results.
  -nd, --no_val_drop              Do not drop last validate batch if if it is
                                  smaller than batch_size.
  -af, --affinity TEXT            Path to affinity matrix for training.
  -cl, --classes TEXT             Path to a CSV file containing a list of
                                  classes for training.
  -clf, --classifier TEXT         Method to classify the latent space. Options
                                  are: KNN (nearest neighbour), NN (neural
                                  network), LR (Logistic Regression).
  -ep, --epochs INTEGER           Number of epochs (default 100).
  -ba, --batch INTEGER            Batch size (default 128).
  -de, --depth INTEGER            Depth of the convolutional layers (default
                                  3).
  -ch, --channels INTEGER         First layer channels (default 64).
  -fl, --filters TEXT             Comma-separated list of filters for the
                                  network. Either provide filters, or capacity
                                  and depth.
  -ld, --latent_dims INTEGER      Latent space dimension (default 10).
  -pd, --pose_dims INTEGER        If pose on, number of pose dimensions. If 0
                                  and gamma=0 it becomesa standard beta-VAE.
  -bn_enc, --bnorm_encoder        Batch normalisation in encoder is on if
                                  True.
  -bn_dec, --bnorm_decoder        Batch normalisation in encoder is on if
                                  True.
  -gsdcl, --gsd_conv_layers INTEGER
                                  The number of output channels for the
                                  convolution layers at the end of the GSD
                                  decoder
  -spl, --n_splats INTEGER        Number of Gaussian splats.
  -kr, --klreduction TEXT         Mean or sum reduction on KLD term.
  -be, --beta FLOAT               Beta maximum in the case of cyclical
                                  annealing schedule
  -bl, --beta_load                The path to the saved beta array file to be
                                  loaded if this file is provided, all other
                                  beta related variables would be ignored
  -g, --gamma FLOAT               Scale factor for the loss component
                                  corresponding to shape similarity. If 0 and
                                  pd=0 it becomes a standardbeta-VAE.
  -gl, --gamma_load               The path to the saved gamma array file to be
                                  loadedif this file is provided, all other
                                  gamma related variables would be ignored
  -lr, --learning FLOAT           Learning rate.
  --recon_loss TEXT               Loss type: 'MSE' or 'BCE' (default 'MSE').
  -bs, --beta_min FLOAT           Beta minimum in the case of cyclical
                                  annealing schedule
  -bc, --beta_cycle INTEGER       Number of cycles for beta during training in
                                  the case of cyclical annealing schedule
  -br, --beta_ratio FLOAT         The ratio for steps in beta
  -cycmb, --cyc_method_beta TEXT  The schedule for : for constant beta : flat,
                                  other options include , cycle_linear,
                                  cycle_sigmoid, cycle_cosine, ramp
  -gs, --gamma_min FLOAT          gamma minimum in the case of cyclical
                                  annealing schedule
  -gc, --gamma_cycle INTEGER      Number of cycles for gamma during training
                                  in the case of cyclical annealing schedule
  -gr, --gamma_ratio FLOAT        The ratio for steps in gamma
  -cycmg, --cyc_method_gamma TEXT
                                  The schedule for gamma: for constant gamma :
                                  flat, other options include , cycle_linear,
                                  cycle_sigmoid, cycle_cosine, ramp
  -g, --gpu                       Use GPU for training.
  --evaluate                      Evaluate test data.
  --vis_dynamic                   Enable collecting meta and dynamic latent
                                  space plots.
  -m, --model TEXT                Choose model to run. The choice of models
                                  are cnn and gsd
  -vl, --vis_los                  Visualise loss (every epoch starting at
                                  epoch 2).
  -vac, --vis_acc                 Visualise confusion matrix and F1 scores
                                  (frequency controlled).
  -vr, --vis_rec                  Visualise reconstructions (frequency
                                  controlled).
  -ve, --vis_emb                  Visualise latent space embedding (frequency
                                  controlled).
  -vi, --vis_int                  Visualise interpolations (frequency
                                  controlled).
  -vt, --vis_dis                  Visualise latent disentanglement (frequency
                                  controlled).
  -vps, --vis_pos                 Visualise pose disentanglement (frequency
                                  controlled).
  -vpsc, --vis_pose_class TEXT    Example: A,B,C. your deliminator should be
                                  commas and no spaces. Classes to be used for
                                  pose interpolation (a seperate pose
                                  interpolation figure would be created for
                                  each class).
  -vzni, --vis_z_n_int TEXT       Number of Latent interpolation classes to to
                                  be printed, number of interpolation steps in
                                  each  plot. Example: 1,10. 1 plot with 10
                                  interpolation steps between two classes.
                                  your deliminator should be commas and no
                                  spaces.
  -vc, --vis_cyc                  Visualise cyclical parameters (once per
                                  run).
  -va, --vis_aff                  Visualise affinity matrix (once per run).
  -his, --vis_his                 Visualise train-val class distribution (once
                                  per run).
  -similarity, --vis_sim          Visualise train-val model similarity matrix.
  -va, --vis_all                  Visualise all above.
  -vf, --vis_format TEXT          The format of saved images. Options: png ,
                                  pdf
  -fev, --freq_eval INTEGER       Frequency at which to evaluate test set.
  -fs, --freq_sta INTEGER         Frequency at which to save state
  -fac, --freq_acc INTEGER        Frequency at which to visualise confusion
                                  matrix.
  -fr, --freq_rec INTEGER         Frequency at which to visualise
                                  reconstructions
  -fe, --freq_emb INTEGER         Frequency at which to visualise the latent
                                  space embedding.
  -fi, --freq_int INTEGER         Frequency at which to visualise latent
                                  spaceinterpolations (default every 10
                                  epochs).
  -ft, --freq_dis INTEGER         Frequency at which to visualise single
                                  transversals.
  -fp, --freq_pos INTEGER         Frequency at which to visualise pose.
  -fsim, --freq_sim INTEGER       Frequency at which to visualise similarity
                                  matrix.
  -fa, --freq_all INTEGER         Frequency at which to visualise all plots
                                  except loss.
  -opt, --opt_method TEXT         The method of optimisation. It can be
                                  adam/sgd/asgd
  -gb, --gaussian_blur            Applying gaussian bluring to the image data
                                  which should help removing noise. The
                                  minimum and maximum for this is hardcoded.
  -nrm, --normalise               Normalise data
  -sftm, --shift_min              Shift the minimum of the data to one zero
                                  and the maximum to one
  -res, --rescale INTEGER         Rescale images to given value (tuple, one
                                  value per dim).
  -tb, --tensorboard              Log metrics and figures to tensorboard
                                  during training
  -st, --strategy TEXT            Define the strategy for distributed
                                  training. Options are: 'ddp', 'deepspeed' or
                                  'fsdp
  --help                          Show this message and exit.
```

Note that setting `-g/--gamma` to `0` and `-pd/--pose_dims` to `0` will run a
vanilla beta-VAE.
</details>


#### Configuring from the command line

You can run on example data with the following command:

```
python affinity-vae/run.py --datapath data/subtomo_files --split 20 --epochs 10 --batch 128 --learning 0.001 --depth 4 --channels 64 --latent_dims 8 --pose_dims 3 --beta 1 --gamma 2 --limit 1000 --freq_all 5 --vis_all --vis_dynamic
```

where the **subtomo_files** is a directory with a number of `.mcr` proteine
image files named with the protein keyword such as
(`1BXN_m0_156_Th0.mrc`,`5MRC_m8_1347_Th0.mrc`, etc). The **subtomo_files**
directory should also have be a `classes.csv` file with a list of the protein
names and keywords to be considered (`1BXN`, `5MRC`, etc.) and a
`affinity_scores.csv` matrix with the initial values for the proteins named in
the `classes.csv`.

#### Using a config submission file

You can also run the code using a submission config file (you can find an
example with default values on `tests/avae-test-config.yml`). For example, you
can run the following command:

```
python affinity-vae/run.py --config_file affinity-vae/tests/avae-test-config.yml
```

You can also use a mix of config file and command line arguments. For example,
you can run the following command:

```
python affinity-vae/run.py --config_file affinity-vae/tests/avae-test-config.yml --epochs 10 --affinity /path/to/different_affinity.csv
```

this will rewrite the values for the epochs and affinity path in the config
file.

At the end of the run, the code will save the final config file used for the run
in the working directory. This will account for any changes made to the config
file from the command line. Running the code again with that config file will
reproduce the results.

In the tools folder you can find notebooks which will assist you in creating the
input files for Affinity-VAE or analyse teh output of the model.

#### Considerations

##### Test folder : If test folder is present, the program will read the test files regardless of the eval flag

##### Evaluation: To run evaluation on a trained model you can turn the `eval` flag to True. This will load the last model present on the `states` directory (within the working directory path where you run the code) and run the evaluation on data set by the `datapath` flag. The evaluation will be saved in the `plots` and `latents` directory with the `eval` suffix on the names.

##### The name of the state file consist of avae_date_time_Epoch_latent_pose.pt

### Inspecting the results from Affinity-VAE

You can interact with the latent space and access reconstruction from the train
model using Napari. You can find more information on how to use the Napari
plugin in the [README.md](scripts/README.md) file in the scripts folder.
