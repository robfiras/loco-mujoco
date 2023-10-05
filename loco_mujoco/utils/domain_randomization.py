import yaml
import mujoco
import numpy as np
from copy import deepcopy
from dm_control import mjcf
from multiprocessing import Queue, Pool


class DomainRandomizationHandler:

    def __init__(self, xml_paths, domain_rand_conf_path, parallel=True, N_worker_per_xml=4):
        assert N_worker_per_xml >= 1 if parallel else True
        self._xml_handles = [mjcf.from_path(f) for f in xml_paths]
        self._domain_rand_conf_path = domain_rand_conf_path
        self._curr_model_id = None
        self.parallel = parallel

        if parallel:
            self._send_queues = [Queue(N_worker_per_xml) for i in range(len(self._xml_handles))]
            self._recv_queues = [Queue(1) for i in range(len(self._xml_handles))]
            self._pools = [Pool(N_worker_per_xml, build_MjModel_from_xml_handle_job,
                               (deepcopy(h), domain_rand_conf_path, sq, rq)) for h, sq, rq in
                           zip(self._xml_handles, self._send_queues, self._recv_queues)]
            for rq in self._recv_queues:
                for i in range(N_worker_per_xml):
                    rq.put("get")

    def get_randomized_model(self, model_id):
        if self.parallel:
            model = self._send_queues[model_id].get()
            self._recv_queues[model_id].put("get")
        else:
            model = build_MjModel_from_xml_handle(self._xml_handles[model_id], self._domain_rand_conf_path)
        return model


def apply_domain_randomization(xml_handle, domain_randomization_config):
    """
    Applies domain/dynamics randomization to the xml_handle based on the provided
    configuration file.

    Args:
        xml_handle: Handle to Mujoco XML.
        domain_randomization_config (str): Path to the configuration file for domain randomization.

    Returns:
        Modified Mujoco XML Handle.

    """

    if domain_randomization_config is not None:
        with open(domain_randomization_config, 'r') as file:
            config = yaml.safe_load(file)
        # apply domain randomization on joints
        if config is not None:
            if "Joints" in config.keys():
                config_joints = config["Joints"]
            else:
                config_joints = None
            if "Default" in config.keys():
                config_default = config["Default"]
            else:
                config_default = None
            all_joints = xml_handle.find_all("joint")
            for jh in all_joints:
                if config_joints is not None and jh.name in config_joints.keys():
                    conf = config_joints[jh.name]
                    set_joint_conf(conf, jh)
                elif config_default is not None and "Joints" in config_default.keys():
                    if "exclude" in config_default.keys() and jh.name not in config_default["exclude"]:
                        conf = config_default["Joints"]
                        set_joint_conf(conf, jh)

    return xml_handle


def set_joint_conf(conf, jh):
    """
    Set the properties of the joint handle (jh) to the randomization properties defined in conf.

    Args:
        conf (dict): Dictionary defining the randomization properties of the joint.
        jh: Mujoco joint handle to be modified.

    Returns:
        Mujoco joint handle.

    """

    for param_name, param in conf.items():
        assert ("sigma" in list(param.keys())) != ("uniform_range" in list(param.keys())), \
            "Specifying sigma and uniform_range on a single joint is not allowed."
        if "sigma" in param.keys():
            if param_name == "damping":
                jh.damping = np.clip(np.random.normal(jh.damping if jh.damping is not None else 0.0,
                                                      param["sigma"]), 0.0, np.Inf)
            elif param_name == "frictionloss":
                jh.frictionloss = np.clip(np.random.normal(jh.frictionloss
                                                           if jh.frictionloss is not None else 0.0,
                                                           param["sigma"]), 0.0, np.Inf)
            elif param_name == "armature":
                jh.armature = np.clip(np.random.normal(jh.armature if jh.armature is not None else 0.0,
                                                       param["sigma"]), 0.0, np.Inf)
            elif param_name == "stiffness":
                jh.stiffness = np.clip(np.random.normal(jh.stiffness
                                                        if jh.stiffness is not None else 0.0,
                                                        param["sigma"]), 0.0, np.Inf)
            else:
                raise ValueError(f"Parameter {param_name} currently nor supported "
                                 f"for domain randomizaiton.")
        elif "uniform_range" in param.keys():
            try:
                low, high = param["uniform_range"]
            except ValueError as e:
                raise Exception(f"The parameter unform_range for {joint_name} is wrongly specified "
                                f"in the domain_randomization_config. The format is:\n"
                                "uniform_range: [low, high]\n") from e
            if param_name == "damping":
                jh.damping = np.random.uniform(low, high)
            elif param_name == "frictionloss":
                jh.frictionloss = np.random.normal(low, high)
            elif param_name == "armature":
                jh.armature = np.random.normal(low, high)
            elif param_name == "stiffness":
                jh.stiffness = np.random.normal(low, high)
            else:
                raise ValueError(f"Parameter {param_name} currently nor supported "
                                 f"for domain randomization.")

    return jh

def build_MjModel_from_xml_handle(xml_handle, path_domain_rand_conf):
    xml_string = xml_handle.to_xml_string()
    xml_assets = xml_handle.get_assets()
    new_xml_handle = mjcf.from_xml_string(xml_string, assets=xml_assets, escape_separators=True)
    new_xml_handle = apply_domain_randomization(new_xml_handle, path_domain_rand_conf)
    model = mujoco.MjModel.from_xml_string(xml=new_xml_handle.to_xml_string(), assets=new_xml_handle.get_assets())
    return model

def build_MjModel_from_xml_handle_job(xml_handle, path_domain_rand_conf, sq, rq):
    while True:
        mess = rq.get()
        if mess == "get":
            model = build_MjModel_from_xml_handle(xml_handle, path_domain_rand_conf)
            sq.put(model)
        elif mess == "kill":
            exit()
        else:
            raise ValueError(f"Unknown message {mess}.")
