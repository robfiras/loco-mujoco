"""Terminal state handlers, which decide whether a state is absorbing and ends the episode."""

from .base import TerminalStateHandler
from .no_terminal import NoTerminalStateHandler
from .height import HeightBasedTerminalStateHandler
from .traj import RootPoseTrajTerminalStateHandler, RootTrackingTerminalStateHandler

# register all terminal state handlers
NoTerminalStateHandler.register()
HeightBasedTerminalStateHandler.register()
RootPoseTrajTerminalStateHandler.register()
RootTrackingTerminalStateHandler.register()
