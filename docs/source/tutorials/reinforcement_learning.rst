Reinforcement Learning
=================================

Even though LocoMuJoCo focuses on imitation learning, it can be also used for plain reinforcement learning. The challenge
here is to define a reward function that produces the desired behavior. Here is a minimal example for defining a reinforcement
learning example:

.. note:: This is for didactic purposes only! It will not produce any useful gait.

.. literalinclude:: ../../../examples/reinforcement_learning/example_unitree_h1.py
    :language: python

Right now, LocoMuJoCo only supports Markovian reward functions (i.e., functions only depending on the current
state transition). We are thinking about providing support for non-Markovian reward functions as well by providing access
to the environment in the reward function. Open an issue or drop me a message if you think this is something
we should really do!

