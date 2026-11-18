"""Decorators used by the environments."""



def info_property(func):
    """
    Decorator to mark a function as an info property.

    This is used to distinguish between regular properties and info of an environment. Info properties are used
    to quickly access specific information about the environment. Its main intended use is inside the LocoEnv and
    all inherited classes.

    Args:
        func (Callable): The getter to turn into a property.

    Returns:
        property: A property whose getter carries the marker attribute ``_is_info_property``.

    """
    if not hasattr(func, '_is_info_property'):
        func._is_info_property = True
    return property(func)
