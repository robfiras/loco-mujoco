from .environments import LocoEnv


def get_all_task_names():
    return LocoEnv.get_all_task_names()


def download_all_datasets():
    return LocoEnv.download_all_datasets()
