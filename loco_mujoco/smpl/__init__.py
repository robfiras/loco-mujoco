"""
Retargeting of human motion capture data to robots through the SMPL-H body model.

Contains the names of the SMPL(-H) joints and of predefined AMASS dataset groups (``const``), the SMPL parsers
(``parser``, only available with the optional ``smpl`` dependencies) and the retargeting pipeline
(``retargeting``).
"""

from .const import *
try:
    from .parser import SMPL_Parser, SMPLH_Parser
except:
    pass
