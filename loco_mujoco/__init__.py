__version__ = '0.1'


from .environments import LocoEnv

try:

    def get_all_task_names():
        return LocoEnv.get_all_task_names()


    def download_all_datasets():
        return LocoEnv.download_all_datasets()

except ImportError:
    pass
