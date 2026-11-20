"""Initial state handlers, which set the state of the simulation at the beginning of an episode."""

from .base import InitialStateHandler
from .default import DefaultInitialStateHandler
from .traj_init_state import TrajInitialStateHandler

# register the initial state handlers
DefaultInitialStateHandler.register()
TrajInitialStateHandler.register()
