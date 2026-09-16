import datetime
import fnmatch
import logging
import os
import pathlib
from logging import config

import pydantic
import yaml

from avae import data


# Model configuration
class AffinityConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", validate_default=True)

    #### Data parameters
    datapath: pydantic.DirectoryPath = pydantic.Field(
        ..., description="Path to data directory"
    )
    affinity: pydantic.FilePath | None = pydantic.Field(
        None, description="Path to affinity file"
    )
    classes: pydantic.FilePath | None = pydantic.Field(
        None, description="Path to classes file"
    )
    datatype: str = pydantic.Field(
        'mrc', pattern='^npy|mrc$', description="Data type"
    )
    limit: pydantic.PositiveInt | None = pydantic.Field(
        None, description="Limit number of samples"
    )
    no_val_drop: bool = pydantic.Field(
        True,
        description="Do not drop last validation batch if is smaller than batch size",
    )
    split: pydantic.PositiveInt = pydantic.Field(20, description="Split ratio")
    date_time_run: str = pydantic.Field(  # TODO remove this
        default_factory=lambda: datetime.datetime.now().strftime(
            "%y%m%d-%H%M"
        ),
        description="Timestamp identifying this run, used in output filenames.",
    )
    new_out: bool = pydantic.Field(
        False, description="Create new output directory"
    )

    #### Pre-processing parameters
    shift_min: bool = pydantic.Field(
        True, description="Scale data with min-max transformation"
    )
    normalise: bool = pydantic.Field(False, description="Normalise data")
    gaussian_blur: bool = pydantic.Field(
        False, description=" Apply gaussian blur"
    )
    rescale: int | None = pydantic.Field(None, description="Rescale data")

    #### Network parameters
    depth: pydantic.PositiveInt = pydantic.Field(
        3, description="Number of layers"
    )
    channels: pydantic.PositiveInt = pydantic.Field(
        64, description="First layer channels"
    )
    filters: list[pydantic.PositiveInt] | None = pydantic.Field(
        None,
        description="Channels per layer, e.g. [8, 16, 32, 64]; overrides channels and depth.",
    )
    latent_dims: pydantic.PositiveInt = pydantic.Field(
        7, description="Latent space dimensions"
    )
    pose_dims: int = pydantic.Field(3, description="Pose dimensions")
    model: str = pydantic.Field(
        'cnn',
        pattern='^(cnn|gsd|u|a|b)$',
        description="Model type: cnn or gsd",
    )
    n_splats: int = pydantic.Field(
        128, description="The number of Gaussian splats for the GSD "
    )
    gsd_conv_layers: int = pydantic.Field(
        0,
        description="If not none, activates convolution layers at the end of the differetiable decoder.",
    )
    bnorm_encoder: bool = pydantic.Field(
        False, description="Use batch normalisation in encoder"
    )
    bnorm_decoder: bool = pydantic.Field(
        False, description="Use batch normalisation in decoder"
    )

    #### Training parameters
    epochs: pydantic.PositiveInt = pydantic.Field(
        100, description="Number of epochs"
    )
    batch: pydantic.PositiveInt = pydantic.Field(16, description="Batch size")
    learning: pydantic.PositiveFloat = pydantic.Field(
        1e-3, description="Learning rate"
    )
    opt_method: str = pydantic.Field(
        'adam',
        description="Optimisation method.It can be adam/sgd/asgd",
        pattern='^(adam|sgd|asgd)$',
    )
    classifier: str = pydantic.Field(
        "NN",
        pattern='^(KNN|NN|LR)$',
        description="Method to classify the latent space. Options "
        "are: KNN (nearest neighbour), NN (neural network), LR (Logistic Regression).",
    )
    gpu: bool = pydantic.Field(True, description="Use GPU")
    gpu_devices: str | None = pydantic.Field(
        None,
        description="Comma-separated CUDA device indices to use (example: 0,1,3).",
    )
    strategy: str = pydantic.Field(
        "auto",
        description="Strategy for training. It can be  'ddp', 'deepspeed' or 'fsdp",
    )
    freq_sta: pydantic.PositiveInt = pydantic.Field(
        10, description="Frequency (in epochs) of states saved."
    )
    freq_eval: pydantic.PositiveInt = pydantic.Field(
        10, description="Frequency (in epochs) of evaluation"
    )

    #### Loss parameters
    beta: float = pydantic.Field(0.1, description="Beta value")
    gamma: float = pydantic.Field(2, description="Gamma value")
    recon_loss: str = pydantic.Field('MSE', description="Loss function")
    klreduction: str = pydantic.Field(
        'mean', description="KL reduction method"
    )

    #### Cyclical annealing parameters for beta
    beta_min: float = pydantic.Field(0, description="Minimum betvalue")
    beta_cycle: pydantic.PositiveInt = pydantic.Field(
        4, description="Beta cycle"
    )
    beta_ratio: pydantic.PositiveFloat = pydantic.Field(
        0.5, description="Beta ratio"
    )
    cyc_method_beta: str = pydantic.Field(
        'flat',
        pattern='^(cycle_sigmoid|flat|cycle_linear|cycle_cosine|ramp)$',
    )
    beta_load: pydantic.FilePath | None = pydantic.Field(
        None, description="Path to beta file"
    )

    gamma_min: float = pydantic.Field(0, description="Minimum gamma value")
    gamma_cycle: pydantic.PositiveInt = pydantic.Field(
        4, description="Gamma cycle"
    )
    gamma_ratio: pydantic.PositiveFloat = pydantic.Field(
        0.5, description="Gamma ratio"
    )
    cyc_method_gamma: str = pydantic.Field(
        'flat',
        pattern='^(cycle_sigmoid|flat|cycle_linear|cycle_cosine|ramp)$',
    )
    gamma_load: pydantic.FilePath | None = pydantic.Field(
        None, description="Path to gamma array file"
    )

    #### Restart (fine-tune) and eval parameters
    evaluate: bool = pydantic.Field(False, description="Evaluation mode")
    restart: bool = pydantic.Field(False, description="Restart training")
    state: pydantic.FilePath | None = pydantic.Field(
        None, description="Path to state file"
    )
    meta: pydantic.FilePath | None = pydantic.Field(
        None, description="Path to meta file"
    )

    #### Visualisation parameters
    # setup params
    vis_format: None | str = pydantic.Field(
        "png", description="The format of saved images. Options: png , pdf"
    )
    tensorboard: bool = pydantic.Field(False, description="Use tensorboard")
    vis_his: bool = pydantic.Field(
        False, description="Visualise class distribution histogram"
    )
    vis_aff: bool = pydantic.Field(
        False, description="Visualise affinity matrix used"
    )
    vis_cyc: bool = pydantic.Field(
        False, description="Visualise beta/gamma cycle function"
    )
    # plotting params
    vis_los: bool = pydantic.Field(True, description="Visualise loss")
    vis_acc: bool = pydantic.Field(True, description="Visualise accuracy")
    vis_rec: bool = pydantic.Field(
        True, description="Visualise reconstruction"
    )
    vis_emb: bool = pydantic.Field(True, description="Visualise latent space")
    vis_dynamic: bool = pydantic.Field(
        True, description="Dynamic visualisation of the latent space"
    )
    vis_sim: bool = pydantic.Field(
        False, description="Visualise latent distance (similarity) matrix"
    )
    vis_int: bool = pydantic.Field(
        False, description="Visualise latent space interpolation"
    )
    vis_z_n_int: None | str = pydantic.Field(
        None,
        description="Visualise latent space interpolation for specific z and n",  # TODO write better explaination
    )
    vis_dis: bool = pydantic.Field(
        False,
        description="Visualise latent space disentanglement for each dimension",
    )
    vis_pos: bool = pydantic.Field(
        False, description="Visualise pose interpolation"
    )
    vis_pose_class: None | str = pydantic.Field(
        None, description="Visualise pose per class interpolation"
    )
    vis_all: bool | None = pydantic.Field(None, description="Visualise all")

    ### Frequency parameters
    freq_acc: pydantic.PositiveInt = pydantic.Field(
        10, description="Frequency (in epochs) of accuracy plot"
    )
    freq_rec: pydantic.PositiveInt = pydantic.Field(
        10, description="Frequency (in epochs) of reconstruction plot"
    )
    freq_emb: pydantic.PositiveInt = pydantic.Field(
        10, description="Frequency (in epochs) of latent space plot"
    )
    freq_dynamic: pydantic.PositiveInt = pydantic.Field(
        10, description="Frequency (in epochs) of dynamic latent space plot"
    )
    freq_sim: pydantic.PositiveInt = pydantic.Field(
        10, description="Frequency (in epochs) of similarity plot"
    )
    freq_int: pydantic.PositiveInt = pydantic.Field(
        10, description="Frequency (in epochs) of interpolation plot"
    )
    freq_dis: pydantic.PositiveInt = pydantic.Field(
        10, description="Frequency (in epochs) of disentanglement plot"
    )
    freq_pos: pydantic.PositiveInt = pydantic.Field(
        10, description="Frequency (in epochs) of pose plot"
    )
    freq_all: pydantic.PositiveInt | None = pydantic.Field(
        None, description="Frequency (in epochs) of all plots"
    )

    #### Setup
    debug: bool = pydantic.Field(False, description="Debug mode")

    @pydantic.model_validator(mode="after")
    def reject_legacy_models_for_new_training(self):
        if self.model in {"u", "a", "b"} and not (
            self.evaluate or self.restart
        ):
            raise ValueError(
                "Models 'u', 'a', and 'b' are supported only for evaluation "
                "or restart of existing checkpoints; use model 'cnn' for "
                "new training."
            )
        return self


def load_config_params(
    config_file: pathlib.Path | None = None,
    sys_args: list | None = None,
    local_args: dict | None = None,
) -> AffinityConfig:
    """
    Load configuration parameters from config file, command line arguments,
    and local arguments (from click).

    System arguments are present to only overwrite config defaults with
    parameters user actually specified on the command line (click parameters
    are prepopulated with config defaults at runtime).

    Parameters
    ----------
    config_file : pathlib.Path | None
        Path to config file.
    sys_args : list | None
        List of system arguments (from command line).
    local_args : dict | None
        Dictionary of command line click arguments - use this option
        in tests when overwriting config.

    Returns
    -------
    data : dict
        Dictionary of configuration parameters.
    """

    logging.info("\n")
    logging.info("############################################### CONFIG")
    logging.info(f"Loading configuration parameters...")

    if local_args is not None and sys_args is None:
        # this scenario only should happen when this function is called in tests
        sys_args = [None] + ['--' + arg for arg in list(local_args.keys())]

    if config_file is None and sys_args is None:
        # params must be provided either through a config file or system arguments
        # at least one param (datapath) must be provided either on command line
        # or in config file
        raise RuntimeError(
            "No configuration file or system arguments provided."
        )

    if sys_args is not None and local_args is None:
        # system parameters were passed without corresponding local arguments
        # to check against
        raise RuntimeError(
            "System arguments provided without corresponding click arguments."
        )

    if config_file is not None:
        # initialise config with config file parameter
        logging.info("Reading submission configuration file %s", config_file)
        with open(config_file, "r") as f:
            loaded_params = yaml.safe_load(f)
            params = loaded_params if isinstance(loaded_params, dict) else {}
    else:
        # if no config file is provided, start from default and update with command line arguments
        params = {}

    # check for command line input values and overwrite config file values
    # we're using sys args because click has populated defaults from confing
    if sys_args is not None:
        assert local_args is not None
        for arg in sys_args[1:]:
            if not arg.startswith("--") or "config_file" in arg:
                continue
            name = arg[2:].split("=")[0]
            if name in local_args.keys():
                # overwrite config file value with command line argument value
                # but only if its on system args (click has defaults)
                if name in params.keys():
                    logging.warning(
                        "Command line argument "
                        + name
                        + " is overwriting config value: "
                        + str(params[name])
                        + " to: "
                        + str(local_args[name])
                    )
                else:
                    logging.info(
                        "Command line argument "
                        + name
                        + " is setting default config value to: "
                        + str(local_args[name])
                    )
                params[name] = local_args[name]

    # validate the config
    try:
        params = AffinityConfig(**params).model_dump()
        logging.info("Config file is valid")
    except pydantic.ValidationError as e:
        logging.info("Config file is invalid:")
        logging.info(e)
        raise RuntimeError("Config file is invalid: " + str(e))

    logging.info("Polishing config parameters...")
    for key, val in params.items():
        # logging.info("%s: %s", key, str(val))
        if type(val) == pathlib.Path:
            # turn relative paths to absolute
            params[key] = str(val.absolute())
        if 'vis' in key and params['vis_all'] is not None:
            # set visualisation to vis_all if it is not set, except for vis_z_n_int and vis_pose_class
            if key in [
                'vis_all',
                'vis_format',
                'vis_z_n_int',
                'vis_pose_class',
            ]:
                continue
            params[key] = params['vis_all']
            logging.warning(
                f"Visualisation parameter 'vis_all' is overriding {key} to {params['vis_all']}"
            )
        if 'freq' in key and params['freq_all'] is not None:
            # set frequency to freq_all if it is not set
            if key in ['freq_all']:
                continue
            params[key] = params['freq_all']
            logging.warning(
                f"Frequency parameter 'freq_all' is overriding {key} to {params['freq_all']}"
            )
        if 'vis' in key and val:
            # make sure frequency is on for associated vis parameters
            if key in [
                'vis_format',
                'vis_his',
                'vis_aff',
                'vis_cyc',
                'vis_los',
                'vis_z_n_int',
                'vis_pose_class',
            ]:
                # these parameters don't have frequency control
                continue
            if params[f"freq_{key.split('_')[-1]}"] == 0:
                freq = 10 if not params['freq_all'] else params['freq_all']
                params[f"freq_{key.split('_')[-1]}"] = freq
                logging.warning(
                    f"Parameter {key} is set to True but its frequency is 0. Setting to {freq} epochs."
                )

    # validate the new config
    try:
        params = AffinityConfig(**params)
        logging.info("Config file is still valid")
    except pydantic.ValidationError as e:
        logging.info("Config file is now invalid:")
        logging.info(e)
        raise RuntimeError("Config file is now invalid: " + str(e))

    # return the validated config object
    return params


def save_config_params(
    params: AffinityConfig, output_dir: pathlib.Path | str = "configs"
) -> pathlib.Path:
    """Save a validated configuration and return the output path."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"config_{params.date_time_run}.yaml"

    with output_path.open("w") as file:
        yaml.safe_dump(params.model_dump(mode="json"), file)

    logging.info("YAML file containing final config saved at: %s", output_path)
    return output_path
