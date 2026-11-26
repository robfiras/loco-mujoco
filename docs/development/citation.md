# Citing LocoMuJoCo

## The library

If you use LocoMuJoCo in your research, please cite the paper on the current version:

```bibtex
@article{alhafez2026locomujoco,
title={LocoMuJoCo: A Comprehensive Imitation Learning Benchmark for Whole-Body Control},
author={Firas Al-Hafez and Davide Tateo and Oleg Kaidanov and Aditya Bhatt and Artur Zelik and Zhengyi Luo and Jan Peters},
journal={arXiv preprint arXiv:2611.26293},
year={2026}
}
```

and the original paper, available at [arXiv:2311.02496](https://arxiv.org/abs/2311.02496). It describes
version 0.1.0; the [changelog](changelog.md) lists what has changed since.

```bibtex
@inproceedings{alhafez2023b,
title={LocoMuJoCo: A Comprehensive Imitation Learning Benchmark for Locomotion},
author={Firas Al-Hafez and Guoping Zhao and Jan Peters and Davide Tateo},
booktitle={6th Robot Learning Workshop, NeurIPS},
year={2023}
}
```

The repository also contains a
[`CITATION.cff`](https://github.com/robfiras/loco-mujoco/blob/master/CITATION.cff) file with both references
and the metadata of the software (version, release date, authors). GitHub shows the current paper under
"Cite this repository", and reference managers such as Zotero can read the file.

To make results reproducible, state the version you used, for example "LocoMuJoCo 1.2.0":

```bash
python -c "import loco_mujoco; print(loco_mujoco.__version__)"
```

## Datasets and models

LocoMuJoCo builds on data and models from other groups. Please also cite the ones your work relies on.

### AMASS

If you use motion from AMASS (`AMASSDatasetConf`), cite AMASS and follow the citation instructions for the
individual sub-datasets on the [AMASS website](https://amass.is.tue.mpg.de).

```bibtex
@inproceedings{mahmood2019amass,
  author    = {Naureen Mahmood and Nima Ghorbani and Nikolaus F. Troje and
               Gerard Pons-Moll and Michael J. Black},
  title     = {{AMASS}: Archive of Motion Capture as Surface Shapes},
  booktitle = {Proceedings of the IEEE/CVF International Conference on
               Computer Vision (ICCV)},
  pages     = {5442--5451},
  year      = {2019},
  doi       = {10.1109/ICCV.2019.00554}
}
```

### LAFAN1

If you use the LAFAN1 clips (`LAFAN1DatasetConf`), cite the
[Ubisoft La Forge animation dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset):

```bibtex
@article{harvey2020lafan1,
  author  = {F{\'e}lix G. Harvey and Mike Yurick and Derek Nowrouzezahrai and
             Christopher Pal},
  title   = {Robust Motion In-Betweening},
  journal = {ACM Transactions on Graphics},
  volume  = {39},
  number  = {4},
  year    = {2020},
  doi     = {10.1145/3386569.3392480}
}
```

### SMPL

The retargeting pipeline fits the [SMPL](https://smpl.is.tue.mpg.de) body model to each robot. If you
retarget motion yourself, which includes all use of AMASS, cite SMPL:

```bibtex
@article{loper2015smpl,
  author  = {Matthew Loper and Naureen Mahmood and Javier Romero and
             Gerard Pons-Moll and Michael J. Black},
  title   = {{SMPL}: A Skinned Multi-Person Linear Model},
  journal = {ACM Transactions on Graphics},
  volume  = {34},
  number  = {6},
  pages   = {248:1--248:16},
  year    = {2015},
  doi     = {10.1145/2816795.2818013}
}
```

### MyoSkeleton and MyoSuite

The `MyoSkeleton` environments use the [MyoSkeleton model](https://github.com/myolab/myo_model) by MyoLab.
Follow the citation instructions in that repository. The general reference for musculoskeletal
simulation with MuJoCo is MyoSuite:

```bibtex
@article{caggiano2022myosuite,
  author  = {Vittorio Caggiano and Huawei Wang and Guillaume Durandau and
             Massimo Sartori and Vikash Kumar},
  title   = {{MyoSuite} -- A Contact-Rich Simulation Suite for Musculoskeletal
             Motor Control},
  journal = {arXiv preprint arXiv:2205.13600},
  year    = {2022},
  doi     = {10.48550/arXiv.2205.13600}
}
```

### MuJoCo, MJX and MuJoCo Warp

All simulation is done by [MuJoCo](https://github.com/google-deepmind/mujoco), by its JAX implementation
MJX, or by [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp):

```bibtex
@inproceedings{todorov2012mujoco,
  author    = {Emanuel Todorov and Tom Erez and Yuval Tassa},
  title     = {{MuJoCo}: A Physics Engine for Model-Based Control},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and
               Systems (IROS)},
  pages     = {5026--5033},
  year      = {2012},
  doi       = {10.1109/IROS.2012.6386109}
}
```

### Trajectory metrics

The trajectory-similarity metrics are computed with [MetrX](https://github.com/pompetzki/metrx):

```bibtex
@misc{pompetzki2024metrx,
  author       = {Kay Pompetzki and Theo Gruner and Firas Al-Hafez and
                  An T. Le and Jan Peters},
  title        = {{MetrX}: A {JAX}-Based Collection of Similarity and
                  Statistical Measures for Accelerated Data Analysis},
  year         = {2024},
  howpublished = {\url{https://github.com/pompetzki/metrx}},
  note         = {Open-source software}
}
```

### Algorithms

If you use one of the included algorithms, cite the paper that introduced it in addition to LocoMuJoCo.

## License

The LocoMuJoCo source code is released under the
[MIT license](https://github.com/robfiras/loco-mujoco/blob/master/LICENSE).

The MIT license covers the code in this repository. Models and data have their own terms:

- **Robot models.** The models in [loco-mujoco-models](https://github.com/robfiras/loco-mujoco-models)
  are third-party models and keep the licenses of their sources. Check that repository before
  redistributing a model.
- **MyoSkeleton.** The model is licensed by MyoLab for non-commercial scientific research.
  `loco-mujoco-myomodel-init` shows the [license](https://github.com/myolab/myo_model/blob/main/LICENSE)
  and downloads the model only after you accept it.
- **SMPL and AMASS.** Both are available from the Max Planck Institute for Intelligent Systems under their
  own license agreements, which you accept when you register and download them. LocoMuJoCo does not
  redistribute them; AMASS motion is retargeted on your machine.
- **LAFAN1 and the default dataset.** The retargeted clips hosted in
  [loco-mujoco-datasets](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets) are derived from
  motion capture data with its own license terms. Check the terms of the original dataset before using
  the clips outside of research.
