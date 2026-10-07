import os.path

import numpy as np
import torch


def as_list(value: object) -> list:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def latest_file(path: str, extension: str) -> str:

    most_recent_file = ""
    most_recent_time = 0

    # iterate over the files in the directory using os.scandir
    for entry in os.scandir(path):
        if entry.name.lower().endswith(extension):
            # get the modification time of the file using entry.stat().st_mtime_ns
            mod_time = entry.stat().st_mtime_ns
            if (
                mod_time > most_recent_time
                and "eval" not in entry.name.lower()
            ):
                # update the most recent file and its modification time
                most_recent_file = entry.name
                most_recent_time = mod_time
    return most_recent_file
