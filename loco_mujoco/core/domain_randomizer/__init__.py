from .base import DomainRandomizer
from .no_randomization import NoDomainRandomization
from .default import DefaultRandomizer
from .prosthesis import ProsthesisRandomizer

# register all domain randomizers
NoDomainRandomization.register()
DefaultRandomizer.register()
ProsthesisRandomizer.register()
