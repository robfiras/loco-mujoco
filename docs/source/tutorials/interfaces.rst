Interfaces
=================================

LocoMuJoCo comes with two different types of interfaces; Gymansium and MushroomRL. LocoMuJoCo natively builds on
MushroomRL, and provides a wrapper for Gymnasium. Both interfaces are very similar and simple to use.
The following examples show how to use these interfaces to create a UnitreeH1 training loop.


Gymnasium
---------------
.. literalinclude:: ../../../examples/simple_gymnasium_env/example_unitree_h1.py
    :language: python

Mushroom-RL
---------------
.. literalinclude:: ../../../examples/simple_mushroom_env/example_unitree_h1.py
    :language: python
