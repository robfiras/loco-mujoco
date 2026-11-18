Trajectory Interface
========================

This example illustrates how to use the trajectory interface in LocoMuJoCo to create and replay a custom motion in the
UnitreeH1 environment. A simple standing trajectory is generated, with a sinusoidal signal applied to the left elbow
joint to introduce movement. The trajectory is constructed by stacking joint positions and computing corresponding
velocities, then packaged using the TrajectoryInfo and TrajectoryData classes. It can be saved, loaded, and replayed
directly in the environment. The example also shows how to pass a custom trajectory to an imitation learning setup
using the ImitationFactory.

.. literalinclude:: ../../../examples/tutorials/10_creating_custom_traj.py
   :language: python
   :linenos:
