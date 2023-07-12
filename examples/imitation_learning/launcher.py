import os
from itertools import product
from experiment_launcher import Launcher
from experiment_launcher.utils import bool_local_cluster


if __name__ == '__main__':
    LOCAL = bool_local_cluster()
    TEST = False
    USE_CUDA = False

    N_SEEDS = 5

    launcher = Launcher(exp_name='loco_mujoco_evalution',
                        python_file='experiment',
                        partition="amd2,amd",
                        n_exps=N_SEEDS,
                        n_cores=3,
                        memory_per_core=1500,
                        days=3,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=True,
                        )

    default_params = dict(n_epochs=300,
                          n_steps_per_epoch=100000,
                          n_epochs_save=25,
                          info_constraint=0.1,
                          n_eval_episodes=10,
                          n_steps_per_fit=1000,
                          discr_only_state=True,
                          use_cuda=USE_CUDA)


    envs = ["Atlas.walk", "Atlas.carry",
            ""]
    lrs = [(1e-4, 5e-5)]
    use_next_states = [0, 1]


    for lr, d, p_ent_coef, use_nt, last_pa, horizon, gamma, dataset_scaling in product(lrs, d_delays, plcy_ent_coefs,
                                                                                       use_noisy_targets, lpa, horizons,
                                                                                       gammas, datasets_scalings):
        lrc, lrD = lr
        expert_data_path, scaling = dataset_scaling
        launcher.add_experiment( last_policy_activation=last_pa, lrc=lrc, lrD__=lrD,
                                use_noisy_targets__=use_nt, horizon__=horizon, gamma__=gamma,
                                expert_data_path=expert_data_path, scaling__=scaling, **default_params)

    launcher.run(LOCAL, TEST)
