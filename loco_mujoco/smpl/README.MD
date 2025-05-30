# Installing the AMASS dependencies

To use the AMASS dataset, you need to download and set up the required data and models. This includes specifying the storage paths for the AMASS dataset and SMPL-H models and ensuring the dependencies are installed. Follow the instructions below to complete the setup.

> [!NOTE]
> this currently works only with **editable installations**!

---

## Additional dependencies

The retargeting pipeline builds on the SMPL models, which require PyTorch. It is recommended to install the CPU version of PyTorch to avoid conflicts with JAX.

To install the CPU version of PyTorch, run:

 ```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
````

Once PyTorch is installed, you can install the rest of the additional dependencies, including SMPL-related packages. 

The SMPL Software Copyright License is for non-commercial scientific research purposes, and you must read and accept the terms and conditions described [here](https://github.com/vchoutas/smplx) before executing the command below. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this [License](https://github.com/vchoutas/smplx/blob/main/LICENSE).

Note that this command will install the GPU version of PyTorch if you haven't explicitly installed the CPU version above:
```bash
pip install -e '.[smpl]'
```

---

## Download the AMASS Dataset and SMPL-H Model

### 1. Setting the Storage Directory
Before downloading the datasets and models, specify the directory where you’d like to store them:



```bash
mkdir <path to folder storing the amass datasets>
loco-mujoco-set-amass-path --path <path to folder storing the amass datasets>
mkdir <path to folder storing the smpl models>
loco-mujoco-set-smpl-model-path --path <path to folder storing the smpl models>
```

For example, to install the datasets in your home folder, you can run:

```bash
mkdir $HOME/amass
loco-mujoco-set-amass-path --path $HOME/amass
mkdir $HOME/smpl
loco-mujoco-set-smpl-model-path --path $HOME/smpl
```

Also, set the path to which the converted datasets (after retargeting) will be stored. If you already specified
a general caching  directory for LocoMuJoCo using `loco-mujoco-set-all-caches`, you can skip this step. But if you want
to store the converted AMASS datasets in a different directory, you can set it using:
```bash
loco-mujoco-set-conv-amass-path --path <path to converted amass data>
```

For example, to set the path into the home directory, you can run:

```bash
loco-mujoco-set-conv-amass-path --path $HOME/amass_conv
```


### 2. Downloading the AMASS Dataset
1. Visit the [AMASS website](https://amass.is.tue.mpg.de/index.html).
2. Register and download the datasets you wish to use.
3. Select the `SMPL-H G` version of each dataset.
4. Extract and place them in the directory specified above, e.g., `$HOME/amass`.

### 3. Downloading the SMPL-H Models
1. Go to the [MANO website](https://mano.is.tue.mpg.de/download.php).
2. Register and download the following:
   - **Extended SMPL+H model** (includes the SMPL-H model w/o hands).
   - **Models & Code** (includes the hand models).
3. Extract and place them in the directory specified above, e.g., `$HOME/smpl`.

---

## Generating the SMPL-H Model

To combine the body and hand models into a single `SMPLH_NEUTRAL.PKL` file, follow one of the methods below:

---

### Automatic Method (Recommended; Requires Anaconda)

To resolve dependency issues with deprecated packages, we provide an automated [script](./install_smplh.sh). This script uses a Conda environment to manage dependencies and ensure proper installation.

1. Ensure [Anaconda](https://www.anaconda.com/) is installed.
2. Copy and paste the [script](./install_smplh.sh) somewhere.
3. Run the script:
   ```bash
   chmod u+x install_smplh.sh   # this is using the original name of the script
   ./install_smplh.sh
   ```

This script will:
- Create a Python 3.10 environment.
- Install all required dependencies.
- Generate the `SMPLH_NEUTRAL.PKL` file and place it in `/path/to/smpl/models`.
- Finally, delete the created Python environment and dependencies. 

---

### Manual Method (Without Conda)

If you don’t have Conda installed or prefer a manual setup, follow these steps. **This method only supports Python 3.10.**

1. **Create a Virtual Environment**:
   ```bash
   python3.10 -m venv smplh_model_conversion_env
   source smplh_model_conversion_env/bin/activate  # On Windows: smplh_model_conversion_env\Scripts\activate
   ```

2. **Install Required Dependencies**:
   ```bash
   pip install 'numpy<1.23.0' chumpy tqdm pyyaml
   ```

3. **Run the Script**:
   ```bash
   cd path/to/loco-mujoco
   python loco_mujoco/smpl/generate_smplh_model.py --smpl-conf-file loco_mujoco/smpl/conf_paths.yaml
   ```

4. **Deactivate the Environment**:
   ```bash
   deactivate
   ```
---

### Test the datasets
You can test the SMPL model by running the script in `examples/replay_datasets/`:

```bash
python smpl_example.py
```

**WARNING!** The script assumes that you have downloaded the `DanceDB` dataset. You can try the script with different datasets by specifying a different dataset file



