import yaml
import mujoco
import numpy as np
from copy import deepcopy
from dm_control import mjcf
from multiprocessing import Queue, Pool


class DomainRandomizationHandler:
    """
    Class for handling domain randomization.

    """

    def __init__(self, xml_handles, domain_rand_conf_path, parallel=True, N_worker_per_xml=4):
        """
        Constructor.

        Args:
            xml_handles : List of Mujoco xml handles.
            domain_rand_conf_path (str): Path to the domain randomization config file.
            parallel (bool): If True, domain randimzation will be done in parallel to speed up the simulation runtime.
            N_worker_per_xml (int): Number of workers for parallel domain randomization.

        """

        assert N_worker_per_xml >= 1 if parallel else True
        self._xml_handles = xml_handles
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
        """ Returns a randomized model based on the model-id. """

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
            if "Inertial" in config.keys():
                config_interial = config["Inertial"]
            else:
                config_interial = None

            # apply all modification to joints
            all_joints = xml_handle.find_all("joint")
            for jh in all_joints:
                if config_joints is not None and jh.name in config_joints.keys():
                    conf = config_joints[jh.name]
                    set_joint_conf(conf, jh)
                elif config_default is not None and "Joints" in config_default.keys():
                    if "exclude" in config_default.keys() and jh.name not in config_default["exclude"]:
                        conf = config_default["Joints"]
                        set_joint_conf(conf, jh)
            # apply all modifications to interial elements
            all_bodies = xml_handle.find_all("body")
            for bh in all_bodies:
                if config_interial is not None and bh.name in config_interial.keys() and bh.inertial is not None:
                    conf = config_interial[bh.name]
                    set_inertial_conf(conf, bh.inertial)
                elif config_default is not None and "Inertial" in config_default.keys() and bh.inertial is not None:
                    conf = config_default["Inertial"]
                    set_inertial_conf(conf, bh.inertial)

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
            low, high = check_uniform_range_conf(jh, param["uniform_range"])
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
        elif "uniform_range_delta":
            found_type = type(param["uniform_range_delta"])
            assert found_type == float, f"uniform_range_delta parameter should be a float, but found {found_type}."

    return jh


def set_inertial_conf(conf, ih):
    """
    Set the properties of the inertial handle (ih) to the randomization properties defined in conf.

    Args:
        conf (dict): Dictionary defining the randomization properties of the inertial.
        ih: Mujoco inertial handle to be modified.

    Returns:
        Mujoco inertial handle.

    """

    for param_name, param in conf.items():
        valid_params = {"sigma", "uniform_range", "uniform_range_delta"}
        found_valid_elements = list(set(param.keys()) & valid_params)    # get number by intersection
        assert len(found_valid_elements) == 1, f"Exactly one parameter should be provided for inertial of " \
                                               f"body {ih.parent.name}, but found {len(found_valid_elements)}" \
                                               f" for {param_name}. Valid parameters are {valid_params}."

        if param_name == "mass":
            assert ih.mass is not None, "Randomizing masses not allowed if not specified in xml."
            if "sigma" in param.keys():
                ih.mass = np.clip(np.random.normal(ih.mass, param["sigma"]), 0.0, np.Inf)
            elif "uniform_range" in param.keys():
                low, high = check_uniform_range_conf(ih, param["uniform_range"])
                ih.mass = np.random.uniform(low, high)
            elif "uniform_range_delta" in param.keys():
                delta = check_uniform_range_delta_conf(ih, param["uniform_range_delta"])
                low, high = ih.mass - delta, ih.mass + delta
                assert low > 0.0, f"uniform_range_delta param ({delta}) for body {ih.parent.name} is bigger" \
                                  f" than mass ({ih.mass}). Negative masses are not allowed."
                ih.mass = np.random.uniform(low, high)

        elif param_name == "diaginertia" or "fullinertia":
            assert "uniform_range_delta" in found_valid_elements, f"domain randomization of inertia only allowed using " \
                                                                  f"uniform_range_delta, but found {list(param.keys())}."
            if param_name == "diaginertia":
                assert ih.diaginertia is not None, "Randomizing diaginertia not allowed if not specified in the xml."
                delta = check_uniform_range_delta_conf(ih, param["uniform_range_delta"])
                lows, highs = ih.diaginertia - delta, ih.diaginertia + delta
                check_lows_singular_values(lows, delta, ih, ih.diaginertia)
                ih.diaginertia = np.random.uniform(lows, highs)
            elif param_name == "fullinertia":
                assert ih.fullinertia is not None, "Randomizing fullinertia not allowed if not specified in the xml."
                delta = check_uniform_range_delta_conf(ih, param["uniform_range_delta"])
                # Do svd and apply randomization only on signular values
                fi = ih.fullinertia
                triu = np.array([[fi[0], fi[3], fi[4]], [0.0, fi[1], fi[5]], [0.0, 0.0, fi[2]]])
                U, sing_val, Vh = np.linalg.svd(triu, compute_uv=True)
                lows, highs = sing_val - delta, sing_val + delta
                check_lows_singular_values(lows, delta, ih, sing_val)
                new_sing_val = np.random.uniform(lows, highs)
                new_triu = U @ np.diag(new_sing_val) @ Vh
                ih.fullinertia = np.array([new_triu[0, 0], new_triu[1, 1], new_triu[2, 2],
                                           new_triu[0, 1], new_triu[0, 2], new_triu[1, 2]])
    return ih


def build_MjModel_from_xml_handle(xml_handle, path_domain_rand_conf):
    """
    Function that takes in an xml_handle and a path to the domain randomization file and returns a randomizaed model.

    Args:
        xml_handle: Mujoco xml handle.
        path_domain_rand_conf (str): Path to the domain randomization file.

    Returns:
        Randomized model.

    """

    new_xml_handle = apply_domain_randomization(xml_handle, path_domain_rand_conf)
    model = mujoco.MjModel.from_xml_string(xml=new_xml_handle.to_xml_string(), assets=new_xml_handle.get_assets())
    return model


def build_MjModel_from_xml_handle_job(xml_handle, path_domain_rand_conf, sq, rq):
    """
    Worker function for parallel domain randomization. It takes in an xml_handle and a path to the domain
    randomization file and puts a randomizaed model in the respective queue.

    Args:
        xml_handle: Mujoco xml handle.
        path_domain_rand_conf (str): Path to the domain randomization file.
        sq (Queue): Send queue used to send the model to the main tread.
        rq (Queue): Receive queue used to receive the trigger to sample another randomized model.

    """

    while True:
        mess = rq.get()
        if mess == "get":
            model = build_MjModel_from_xml_handle(xml_handle, path_domain_rand_conf)
            sq.put(model)
        elif mess == "kill":
            exit()
        else:
            raise ValueError(f"Unknown message {mess}.")


def check_uniform_range_conf(h, params, check_low_greater_zero=True):
    if hasattr(h, "name"):
        name = h.name
    else:
        name = h.parent.name

    try:
        low, high = params
    except ValueError as e:
        raise Exception(f"The parameter uniform_range for {name} is wrongly specified "
                        f"in the domain_randomization_config. The format is:\n"
                        "uniform_range: [low, high]\n") from e
    assert high > low, f"uniform_range for body {name} wrongly specified, because high < low"
    if check_low_greater_zero:
        assert low >= 0.0, f"uniform_range for body {name} wrongly specified, because low < 0.0"
    return low, high


def check_uniform_range_delta_conf(h, params):
    if hasattr(h, "name"):
        name = h.name
    else:
        name = h.parent.name

    found_type = type(params)
    assert found_type == float, f"uniform_range_delta parameter for {name} should be a float, but found {found_type}."
    delta = params
    return delta


def check_lows_singular_values(lows, delta, ih, sing_val):
    assert np.all(lows > 0.0), f"Error for body f{ih.parent.name}. " \
                               f"uniform_range_delta param ({delta}) is bigger than the smallest" \
                               f"singular values ({np.min(sing_val)}). " \
                               f"Negative singular values are not allowed."