Rendering in the Browser and Exporting to USD
==============================================

Next to the OpenGL viewer used by ``render()`` and ``mjx_render()``, every environment has a
`viser <https://viser.studio>`__-based counterpart that serves the scene to a browser instead of
opening a local window. This is useful over SSH and on machines without a display. It requires
the optional dependencies:

.. code-block:: bash

   pip install -e ".[viser]"

The server URL (``http://localhost:8080`` by default) is printed on startup. It can be configured
with the ``host``, ``port`` and ``verbose`` environment arguments.

Rendering a MuJoCo Environment
-------------------------------

.. literalinclude:: ../../../examples/tutorials/12_viser_rendering.py
   :language: python
   :linenos:

Rendering Parallel MJX Environments
------------------------------------

``mjx_render_viser()`` lays the parallel environments out on the same grid as ``mjx_render()`` and
adds an environment selector to the browser panel.

.. literalinclude:: ../../../examples/tutorials/13_viser_rendering_mjx.py
   :language: python
   :linenos:

Exporting to USD
-----------------

CPU and MJX simulation states can also be collected into a portable animated USD stage:

.. code-block:: bash

   pip install -e ".[usd]"

.. code-block:: python

   env.play_trajectory(
       n_episodes=1,
       usd=True,
       usd_params={"output_directory": "render", "fps": 30},
   )

More details on the viser appearance options and the USD export are available in the
`core README <https://github.com/robfiras/loco-mujoco/tree/dev/loco_mujoco/core>`__.
