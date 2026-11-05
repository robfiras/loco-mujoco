.. _amass_installation:

Installing AMASS Dependencies
=================================

To use the AMASS dataset, you need to download and set up the required data and models.
This includes specifying storage paths for the AMASS dataset and SMPL-H models and installing the necessary dependencies.

Follow the instructions below to complete the setup.

.. note::

   This currently works only with **editable installations**.

Additional Dependencies
-----------------------

The retargeting pipeline builds on the SMPL models, which require **PyTorch**.
It is recommended to install the **CPU version** of PyTorch to avoid conflicts with **JAX**.

To install the CPU version:

.. code-block:: bash

   pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

Then, install the SMPL-related packages (note: this may install the GPU version of PyTorch unless you've already installed the CPU version):

.. code-block:: bash

   pip install -e '.[smpl]'

Download the AMASS Dataset and SMPL-H Model
-------------------------------------------

1. Setting the Storage Directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create the directories and set the paths:

.. code-block:: bash

   mkdir <path to folder storing the amass datasets>
   loco-mujoco-set-amass-path --path <path to folder storing the amass datasets>
   mkdir <path to folder storing the smpl models>
   loco-mujoco-set-smpl-model-path --path <path to folder storing the smpl models>

Example using the home directory:

.. code-block:: bash

   mkdir $HOME/amass
   loco-mujoco-set-amass-path --path $HOME/amass
   mkdir $HOME/smpl
   loco-mujoco-set-smpl-model-path --path $HOME/smpl

To set the path for converted AMASS datasets (optional if ``loco-mujoco-set-all-caches`` was used):

.. code-block:: bash

   loco-mujoco-set-conv-amass-path --path <path to converted amass data>

Example:

.. code-block:: bash

   loco-mujoco-set-conv-amass-path --path $HOME/amass_conv

2. Downloading the AMASS Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Visit the `AMASS website <https://amass.is.tue.mpg.de/index.html>`_
- Register and download the datasets you wish to use.
- Select the **SMPL-H G** version.
- Extract the data and place it in the directory you created (e.g., ``$HOME/amass``).

3. Downloading the SMPL-H Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Go to the `MANO website <https://mano.is.tue.mpg.de/download.php>`_
- Register and download:

  - **Extended SMPL+H model** (body model)
  - **Models & Code** (hand models)

- Extract and place in the appropriate directory (e.g., ``$HOME/smpl``). The conversion below expects
  ``smplh/neutral/model.npz``, ``mano_v1_2/models/MANO_LEFT.pkl``, and ``mano_v1_2/models/MANO_RIGHT.pkl`` in this directory.

Generating the SMPL-H Model
---------------------------

To combine the body and hand models into a single ``SMPLH_NEUTRAL.pkl`` file, use one of the methods below.

Automatic Method (Recommended; Requires Anaconda)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To resolve dependency issues, use the provided automated script (``loco_mujoco/smpl/install_smplh.sh``).
It creates a Conda environment to isolate the build.

1. Ensure `Anaconda <https://www.anaconda.com/>`_ is installed.
2. Copy the ``install_smplh.sh`` script to a location of your choice.
3. Run the script:

.. code-block:: bash

   chmod u+x install_smplh.sh
   ./install_smplh.sh

This will:

- Create a Python 3.10 Conda environment
- Install dependencies
- Generate ``SMPLH_NEUTRAL.pkl`` in your SMPL model directory
- Clean up the temporary environment

Manual Method (Without Conda)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you do not use Conda, follow this method (only supports **Python 3.10**):

1. Create a virtual environment:

.. code-block:: bash

   python3.10 -m venv smplh_model_conversion_env
   source smplh_model_conversion_env/bin/activate

   # On Windows:
   # smplh_model_conversion_env\Scripts\activate

2. Install required dependencies:

.. code-block:: bash

   pip install 'numpy<1.23.0' chumpy tqdm pyyaml

3. Run the conversion script:

.. code-block:: bash

   cd path/to/loco-mujoco
   python loco_mujoco/smpl/generate_smplh_model.py --smpl-conf-file loco_mujoco/LOCOMUJOCO_VARIABLES.yaml

4. Deactivate the environment:

.. code-block:: bash

   deactivate

Testing the Datasets
--------------------

To test your SMPL model setup, run the provided example:

.. code-block:: bash

   python examples/replay_datasets/smpl_example.py

.. warning::

   This script assumes that you have downloaded the ``DanceDB`` dataset.
   You can test other datasets by modifying the script to specify a different dataset file.
