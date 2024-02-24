.. _dom-rand-tutorial:

Domain Randomization
=================================

In this tutorial, we will show how to use the domain randomization feature. This feature is useful to train a
robot to be robust to changes in the environment, such as joint friction, mass, or inertia. Before starting, make sure
to get familiar with the :ref:`dom-rand`, where you find a detailed documentation.

Consider the following domain randomization file for the Talos robot:

.. literalinclude:: ../../../examples/domain_randomization/domain_randomization_talos.yaml
    :language: yaml

Once a configuration file is created, we can pass it to the environment and start training as usual.
Here is an example of how to use the domain randomization feature with the Talos robot:

.. literalinclude:: ../../../examples/domain_randomization/example_talos.py
    :language: python

.. note:: We provide more examples in respective directory in the main LocoMuJoCo repository.
