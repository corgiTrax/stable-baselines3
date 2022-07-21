import base64
import functools
import io
import json
import os
import pathlib
import pickle
import warnings
import zipfile
from typing import Any, Dict, Optional, Tuple, Union

import cloudpickle
import torch as th

import stable_baselines3 as sb3
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device, get_system_info
from stable_baselines3.common.save_util import *

def load_from_zip_file(
    load_path: Union[str, pathlib.Path, io.BufferedIOBase],
    load_data: bool = True,
    custom_objects: Optional[Dict[str, Any]] = None,
    device: Union[th.device, str] = "auto",
    verbose: int = 0,
    print_system_info: bool = False,
) -> (Tuple[Optional[Dict[str, Any]], Optional[TensorDict], Optional[TensorDict]]):
    """
    Load model data from a .zip archive

    :param load_path: Where to load the model from
    :param load_data: Whether we should load and return data
        (class parameters). Mainly used by 'load_parameters' to only load model parameters (weights)
    :param custom_objects: Dictionary of objects to replace
        upon loading. If a variable is present in this dictionary as a
        key, it will not be deserialized and the corresponding item
        will be used instead. Similar to custom_objects in
        ``keras.models.load_model``. Useful when you have an object in
        file that can not be deserialized.
    :param device: Device on which the code should run.
    :param verbose: Verbosity level, 0 means only warnings, 2 means debug information.
    :param print_system_info: Whether to print or not the system info
        about the saved model.
    :return: Class parameters, model state_dicts (aka "params", dict of state_dict)
        and dict of pytorch variables
    """
    load_path = open_path(load_path, "r", verbose=verbose, suffix="zip")

    # set device to cpu if cuda is not available
    device = get_device(device=device)

    # Open the zip archive and load data
    try:
        with zipfile.ZipFile(load_path) as archive:
            namelist = archive.namelist()
            print(namelist)
            # If data or parameters is not in the
            # zip archive, assume they were stored
            # as None (_save_to_file_zip allows this).
            data = None
            pytorch_variables = None
            params = {}

            # Debug system info first
            if print_system_info:
                if "system_info.txt" in namelist:
                    print("== SAVED MODEL SYSTEM INFO ==")
                    print(archive.read("system_info.txt").decode())
                else:
                    warnings.warn(
                        "The model was saved with SB3 <= 1.2.0 and thus cannot print system information.",
                        UserWarning,
                    )

            if "data" in namelist and load_data:
                # Load class parameters that are stored
                # with either JSON or pickle (not PyTorch variables).
                json_data = archive.read("data").decode()
                data = json_to_data(json_data, custom_objects=custom_objects)

            # Check for all .pth files and load them using th.load.
            # "pytorch_variables.pth" stores PyTorch variables, and any other .pth
            # files store state_dicts of variables with custom names (e.g. policy, policy.optimizer)
            pth_files = [
                file_name
                for file_name in namelist
                if os.path.splitext(file_name)[1] == ".pth"
            ]
            # print(pth_files)
            for file_path in pth_files:
                with archive.open(file_path, mode="r") as param_file:
                    # File has to be seekable, but param_file is not, so load in BytesIO first
                    # fixed in python >= 3.7
                    file_content = io.BytesIO()
                    file_content.write(param_file.read())
                    # go to start of file
                    file_content.seek(0)
                    # Load the parameters with the right ``map_location``.
                    # Remove ".pth" ending with splitext
                    th_object = th.load(file_content, map_location=device)
                    # "tensors.pth" was renamed "pytorch_variables.pth" in v0.9.0, see PR #138
                    if (
                        file_path == "pytorch_variables.pth"
                        or file_path == "tensors.pth"
                    ):
                        # PyTorch variables (not state_dicts)
                        pytorch_variables = th_object
                    else:
                        # State dicts. Store into params dictionary
                        # with same name as in .zip file (without .pth)
                        params[os.path.splitext(file_path)[0]] = th_object
    except zipfile.BadZipFile:
        # load_path wasn't a zip file
        raise ValueError(f"Error: the file {load_path} wasn't a zip-file")
    print([i for i in params['policy']])
    # print(pytorch_variables)

    return data, params, pytorch_variables

# load_from_zip_file(load_path='trained-models/realrobot_ballbasket/ActiveTamerRLSACOptimBallBasket_1000.pt')
load_from_zip_file(load_path='models/robot_task1/robot_active/subject2/ActiveTamerRLSACRecordStop_900.pt')