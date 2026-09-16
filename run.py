import logging
import os
import pathlib
import sys
import types
import typing
import warnings

import click

import avae.config as config
from avae.evaluate import evaluate
from avae.train import train


def click_from_config(model_cls):
    """Generate one click option per model field, using the model as the
    single source of truth for names, help and existence. All options default
    to None so precedence stays: model defaults < config file < CLI."""

    def decorator(fn):
        for name, field in reversed(list(model_cls.model_fields.items())):
            params = {
                "type": None,
                "help": None,
                "show_default": True,
            }

            # first resolve multiple types
            # print("    Name:", name)
            # print("    Field: ", field)
            # print("    Field.annotation:", field.annotation)
            # print("    typing.get_origin(fieldname.annotation):", typing.get_origin(field.annotation))
            # print("    Field.metadata", field.metadata)
            if typing.get_origin(field.annotation) in (
                typing.Union,
                types.UnionType,
            ):
                # check for dual types (e.g. pathlib.Path | None) and extract the non-None type
                annotation = next(
                    a
                    for a in typing.get_args(field.annotation)
                    if a is not type(None)
                )
                # print("        Annotation: ", annotation)
                if typing.get_origin(annotation) is typing.Annotated:
                    annotated_args = typing.get_args(annotation)
                    annotation = annotated_args[0]
                    pathlib_metadata = (
                        annotated_args[1:]
                        if annotation is pathlib.Path
                        else []
                    )
            else:
                annotation = field.annotation
                pathlib_metadata = (
                    field.metadata if annotation is pathlib.Path else []
                )

            # print("        Path?", annotation is pathlib.Path)
            # then set type
            if annotation is pathlib.Path and pathlib_metadata:
                for item in pathlib_metadata:
                    if getattr(item, "path_type", None) is not None:
                        path_type = item.path_type
                        break
                if path_type == "dir":
                    params["type"] = click.Path(
                        path_type=pathlib.Path, dir_okay=True, file_okay=False
                    )
                elif path_type == "file":
                    params["type"] = click.Path(
                        path_type=pathlib.Path, dir_okay=False, file_okay=True
                    )
                else:
                    params["type"] = click.Path(path_type=pathlib.Path)
            else:
                params["type"] = annotation

            # then set the remaining parameters
            if annotation is bool:
                params["is_flag"] = True
            if field.description is not None:
                params["help"] = field.description
            if not field.is_required() and field.default is not None:
                params["default"] = field.default

            names = [f"--{name}"]
            # print(names, params)
            # print()
            fn = click.option(*names, **params)(fn)
        return fn

    return decorator


@click.command(name="Affinity VAE")
@click.option(
    '--config_file',
    type=click.Path(path_type=pathlib.Path, dir_okay=False, file_okay=True),
    default=None,
    help="Path to config file",
)
@click_from_config(config.AffinityConfig)
def run(**kwargs):

    warnings.simplefilter("ignore", FutureWarning)

    # read config file and command line arguments and parse
    params = config.load_config_params(
        config_file=kwargs.get("config_file"),
        sys_args=sys.argv,
        local_args=kwargs,
    )

    if params.debug:
        logging.info("Debug mode enabled")
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("matplotlib.font_manager").disabled = True

    if params.new_out:
        dir_name = f'results_{params.date_time_run}_model_{params.model}_lat{params.latent_dims}_pose{params.pose_dims}_lr{params.learning}_beta{params.beta}_gamma{params.gamma}'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        else:
            logging.info(f"Directory {dir_name} already exists")
        os.chdir(dir_name)

    config.save_config_params(params)

    if not os.path.exists("logs"):
        os.mkdir("logs")

    # setup logger inside the  directory where we are running the code
    fileh = logging.FileHandler(
        "logs/avae_run_log_" + params.date_time_run + ".log", "a"
    )
    logging.getLogger().addHandler(fileh)

    try:
        run_pipeline(params)

    except Exception as e:
        logging.exception("An exception was thrown: %s", e)
        raise


def run_pipeline(params):

    if not params.evaluate:
        if (
            params.affinity is None
        ):  # TODO change to no affinity allowed when training with contrastive loss+ look for it in the basic dir and move it to train
            logging.info("Affinity not provided.")
            raise RuntimeError(
                "Affinity (--affinity) not provided. Please provide affinity to train the model."
            )
        train(params)
    else:
        evaluate(params)


if __name__ == "__main__":
    run()
