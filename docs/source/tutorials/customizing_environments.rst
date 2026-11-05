Customizing Environments
=========================

Within this tutorial, we show how to easily customize environments to your need. While all environments come with default
modules -- like the observations, the control type, or terrain --, any module of the
LocoMuJoCo core can be changed or even defined by yourself. Below, you find some examples.


Custom Observation Space
---------------------------

Within this example, we change the observation space of the environment. The observation specification is passed as a list of
different observation objects. All observation types can be found :mod:`here <loco_mujoco.core.observations.base>`.
The ``create_observation_summary`` function can be used to get a detailed overview over the size and indices of the created
observation space in the browser.

.. literalinclude:: ../../../examples/tutorials/05_changing_the_observation_space.py
   :language: python
   :linenos:


Observation Space Grouping
---------------------------

Often it is required to create different observations for different parts of your algorithm. To do so,
observations can be assigned to one or more groups. Then group masks can be used to mask out observation
only relevant for the respective group. One popular use case is to create observations with limited information
for the policy while passing much more detailed or noiseless observations to the critic. Custom observations can
be also implemented as shown in the last example below.

.. literalinclude:: ../../../examples/tutorials/06_changing_the_observation_space_grouping.py
   :language: python
   :linenos:

Control Types
---------------------------

LocoMuJoCo offers a flexible interface for configuring different control types. While some robots use
torque control by default, others rely on position control. You can easily switch between control types
as needed. In the example below, we demonstrate how to change the control type from torque to position control.

.. literalinclude:: ../../../examples/tutorials/07_changing_control_type.py
   :language: python
   :linenos:


Domain Randomization
---------------------------

Domain randomization is the go-to solution for sim-to-real robotics. We provide an interface for
arbitrary randomization of MuJoCo's MjModel (respectively for Mjx), MjData (respectively for Mjx), the observations and the actions. Below,
you can find an example on how to use the default domain randomizer (classical one for locomotion). You can also define a
completely custom domain randomization modules using our interface as shown below.

.. literalinclude:: ../../../examples/tutorials/08_domain_randomization.py
   :language: python
   :linenos:


Terrains
---------------------------

We also provide a terrain interface allowing to define custom terrain types. For now,
we only support flat and rough terrains, but any custom terrain can be implemented by following
the interface. In the example below, we show how to change to the rough terrain. Note that
the free joint xy-position of the robot is reset to the origin once the border is reached.
All other parts of the simulation state remain unchanged.

.. literalinclude:: ../../../examples/tutorials/09_terrain.py
   :language: python
   :linenos:


Creating Custom Modules
---------------------------

All :ref:`core modules <core_modules>` can be fully customized by the user.
Below is an example showcasing how to define and register a custom environment along
with a custom initial state handler, control function, reward, and observation modules.
Once registered, these components can be reused across different environments, enabling
a highly modular and flexible setup in LocoMuJoCo.

.. literalinclude:: ../../../examples/tutorials/11_creating_custom_modules.py
   :language: python
   :linenos:

