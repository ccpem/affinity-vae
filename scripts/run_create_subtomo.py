import datetime
import logging
import os
import warnings

import click
import yaml

from tools.create_subtomo import create_subtomo

if not os.path.exists("../logs"):
    os.mkdir("../logs")
dt_name = datetime.datetime.now().strftime("%H_%M_%d_%m_%Y")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/avae_run_log_" + dt_name + ".log"),
        logging.StreamHandler(),
    ],
)


class ConvertStrToList(click.Option):
    def type_cast_value(self, ctx, value) -> list:
        try:
            value = str(value)
            assert value.count("[") == 1 and value.count("]") == 1
            list_as_str = value.replace('"', "'").split("[")[1].split("]")[0]
            list_of_items = [
                item.strip().strip("'") for item in list_as_str.split()
            ]
            return list_of_items
        except Exception:
            raise click.BadParameter(value)


@click.command(name="Subtomogram creator")
@click.option("--config_file", type=click.Path(exists=True))
@click.option(
    "--input_path",
    "-ip",
    type=str,
    default=None,
    help="Path to the folder containing the full tomogram/image.",
)
@click.option(
    "--output_path",
    "-op",
    type=str,
    default=None,
    help="Path to the folder for output subtomograms",
)
@click.option(
    "--annot_path",
    "-op",
    type=str,
    default=None,
    help="path to the folder containing the name of the particles and their x,y,z coordinates",
)
@click.option(
    "--datatype",
    "-dtype",
    type=str,
    default=None,
    help="Type of the data: mrc, npy",
)
@click.option(
    "--vox_size",
    "-vs",
    cls=ConvertStrToList,
    default=[],
    help="size of each subtomogram voxel given as a list where vox_size: [x,y,x]",
)
@click.option(
    "--bandpass",
    "-bp",
    type=bool,
    default=None,
    is_flag=True,
    help="Apply band pass",
)
@click.option(
    "--low_freq",
    "-lf",
    type=float,
    default=None,
    help="Lower  frequency threshold for the band pass filter",
)
@click.option(
    "--high_freq",
    "-hf",
    type=float,
    default=None,
    help="higher frequency threshold for the band pass filter",
)
@click.option(
    "--gaussian_blur",
    "-gb",
    type=bool,
    default=None,
    is_flag=True,
    help="Applying gaussian bluring to the image data which should help removing noise. The minimum and maximum for this is hardcoded.",
)
@click.option(
    "--normalise",
    "-nrm",
    type=bool,
    default=None,
    is_flag=True,
    help="Normalise data",
)
@click.option(
    "--add_noise",
    "-n",
    type=bool,
    default=None,
    is_flag=True,
    help="Add noise to images, this can be used for benchmarking",
)
@click.option(
    "--noise_int",
    "-ni",
    type=int,
    default=None,
    help="noise intensity",
)
@click.option(
    "--padding",
    "-ni",
    type=list,
    default=None,
    help="size of padding boxes",
)
@click.option(
    "--augment",
    "-aug",
    type=int,
    default=None,
    help="perform the given number of aumentation on each voxel before saving",
)
@click.option(
    "--aug_th_min",
    "-atmin",
    type=int,
    default=None,
    help="The minimum value of the augmentation range in degrees",
)
@click.option(
    "--aug_th_max",
    "-atmax",
    type=int,
    default=None,
    help="The maximum value of the augmentation range in degrees",
)
def run(
    config_file,
    input_path,
    output_path,
    datatype,
    annot_path,
    vox_size,
    bandpass,
    low_freq=0,
    high_freq=15,
    gaussian_blur=False,
    normalise=False,
    add_noise=False,
    noise_int=0,
    padding=None,
    augment=False,
    aug_num=5,
    aug_th_min=-45,
    aug_th_max=45,
):

    warnings.simplefilter("ignore", FutureWarning)
    # read config file and command line arguments and assign to local variables that are used in the rest of the code
    local_vars = locals().copy()
    print(local_vars)

    if config_file is not None:
        with open(config_file, "r") as f:
            logging.info("Reading submission configuration file" + config_file)
            data = yaml.load(f, Loader=yaml.FullLoader)
        # returns JSON object as

        for key, val in local_vars.items():
            if (
                val is not None
                and isinstance(val, (int, float, bool, str))
                or data.get(key) is None
            ):
                logging.warning(
                    "Command line argument "
                    + key
                    + " is overwriting config file value to: "
                    + str(val)
                )
                data[key] = val
            else:
                logging.info(
                    "Setting "
                    + key
                    + " to config file value: "
                    + str(data[key])
                )
    else:
        # if no config file is provided, use command line arguments
        data = local_vars

        logging.info(
            "Saving final submission config file to: "
            + "avae_final_config"
            + dt_name
            + ".yaml"
        )
        file = open("avae_final_config" + dt_name + ".yaml", "w")
        yaml.dump(data, file)
        file.close()
        logging.info("YAML File saved!")

    create_subtomo(
        input_path=data["input_path"],
        output_path=data["output_path"],
        datatype=data["datatype"],
        annot_path=data["annot_path"],
        vox_size=data["vox_size"],
        bandpass=data["bandpass"],
        low_freq=data["low_freq"],
        high_freq=data["high_freq"],
        gaussian_blur=data["gaussian_blur"],
        normalise=data["normalise"],
        add_noise=data["add_noise"],
        noise_int=data["noise_int"],
        padding=data["padding"],
        augment=data["augment"],
        aug_th_min=data["aug_th_min"],
        aug_th_max=data["aug_th_max"],
    )


if __name__ == "__main__":
    run()
