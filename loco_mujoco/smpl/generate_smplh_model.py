"""
Script that creates the neutral SMPL-H model file used by the retargeting pipeline.

Run with ``python generate_smplh_model.py --smpl-conf-file <path to LOCOMUJOCO_VARIABLES.yaml>``.
"""

import os.path as osp
import pickle
import argparse

import yaml
import numpy as np


def generate_smplh_model(path_to_conf):
    """
    Merge the SMPL-H body model and the MANO hand models into a single model file.

    The neutral SMPL-H body model (``smplh/neutral/model.npz``) and the hand components, coefficients and means of the
    left and right MANO models (``mano_v1_2/models/MANO_LEFT.pkl`` and ``MANO_RIGHT.pkl``) are read from the SMPL
    model directory and written to ``SMPLH_NEUTRAL.pkl`` in the same directory.

    Args:
        path_to_conf (str): Path to the YAML file with the LocoMuJoCo variables. The SMPL model directory is read
            from its entry ``LOCOMUJOCO_SMPL_MODEL_PATH``.

    """

    # read paths from yaml file
    try:
        with open(path_to_conf, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            path_to_smpl_model = data["LOCOMUJOCO_SMPL_MODEL_PATH"]
    except Exception:
        print(f"\nCould not load the smpl conf file. Please use the command "
              f"`loco-mujoco-set-smpl-model-path --path \"/path/to/smpl/models\"` to set the path.")
        raise

    path_to_smplh = path_to_smpl_model + "/smplh/neutral/model.npz"
    path_to_mano_left = path_to_smpl_model + "/mano_v1_2/models/MANO_LEFT.pkl"
    path_to_mano_right = path_to_smpl_model + "/mano_v1_2/models/MANO_RIGHT.pkl"

    # load smplh model
    body_data_np = np.load(path_to_smplh)
    body_data = {}
    for key in body_data_np:
        body_data[key] = body_data_np[key]

    # load hands
    with open(path_to_mano_left, 'rb') as lhand_file:
        lhand_data = pickle.load(lhand_file, encoding='latin1')

    with open(path_to_mano_right, 'rb') as rhand_file:
        rhand_data = pickle.load(rhand_file, encoding='latin1')

    model_name = "SMPLH_NEUTRAL.pkl"

    # combine
    output_data = body_data.copy()
    output_data['hands_componentsl'] = lhand_data['hands_components']
    output_data['hands_componentsr'] = rhand_data['hands_components']

    output_data['hands_coeffsl'] = lhand_data['hands_coeffs']
    output_data['hands_coeffsr'] = rhand_data['hands_coeffs']

    output_data['hands_meanl'] = lhand_data['hands_mean']
    output_data['hands_meanr'] = rhand_data['hands_mean']

    for key, data in output_data.items():
        if 'chumpy' in str(type(data)):
            output_data[key] = np.array(data)
        else:
            output_data[key] = data

    # dump
    out_path = osp.join(path_to_smpl_model, model_name)
    print('Saving to {}'.format(out_path))
    with open(out_path, 'wb') as output_file:
        pickle.dump(output_data, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smpl-conf-file', dest='smpl_conf_file', required=True,
                        type=str, help='The path to the SMPLH model')
    smpl_conf_file = parser.parse_args().smpl_conf_file
    generate_smplh_model(smpl_conf_file)
