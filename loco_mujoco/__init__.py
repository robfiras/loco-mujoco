__version__ = '0.3.0'

try:

    from .environments import LocoEnv

    def get_all_task_names():
        return LocoEnv.get_all_task_names()

except ImportError as e:
    print(e)
