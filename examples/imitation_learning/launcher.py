from itertools import product
from experiment_launcher import Launcher
from experiment_launcher.utils import is_local


if __name__ == '__main__':
    LOCAL = is_local()
    TEST = False
    USE_CUDA = False

    N_SEEDS = 1

    launcher = Launcher(exp_name='loco_mujoco_evalution',
                        exp_file='experiment',
                        partition="amd2,amd",
                        n_seeds=N_SEEDS,
                        n_cores=3,
                        memory_per_core=1500,
                        days=3,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        use_timestamp=True,
                        )

    default_params = dict(n_epochs=1,
                          n_steps_per_epoch=100000,
                          n_epochs_save=25,
                          info_constraint=0.1,
                          n_eval_episodes=10,
                          n_steps_per_fit=1000,
                          discr_only_state=True,
                          use_cuda=USE_CUDA)

    envs = ["Atlas.walk", "Atlas.carry",
            "HumanoidTorque.walk", "HumanoidTorque.run",
            "HumanoidTorque4Ages.walk.1", "HumanoidTorque4Ages.walk.2",
            "HumanoidTorque4Ages.walk.3", "HumanoidTorque4Ages.walk.4", "HumanoidTorque4Ages.walk.all",
            "HumanoidTorque4Ages.run.1", "HumanoidTorque4Ages.run.2",
            "HumanoidTorque4Ages.run.3", "HumanoidTorque4Ages.run.4", "HumanoidTorque4Ages.run.all",
            "UnitreeA1.simple", "UnitreeA1.hard"]

    lrs = [(1e-4, 5e-5), ]
    use_next_statess = [False]

    for env, lr, use_nt in product(envs, lrs, use_next_statess):
        lrc, lrD = lr
        launcher.add_experiment(env__=env, lrc=lrc, lrD=lrD, use_next_states=use_nt, **default_params)

    launcher.run(LOCAL, TEST)
