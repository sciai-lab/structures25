<div align="center">

# SCIAI-DFT

[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.4.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.5-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>

</div>

![figure1](https://github.com/user-attachments/assets/00abb696-95e3-4aaa-857b-2b7548d45646)

This repository contains the code used in our publication [Stable and Accurate Orbital-Free Density Functional Theory Powered by Machine Learning](https://pubs.acs.org/doi/10.1021/jacs.5c06219). Using equivariant graph neural networks, we enable Orbital-Free Density Functional Theory calculations by learning the kinetic energy functional from data.

## Installation

### UV (recommended)

For installation with CUDA run

```bash
uv sync
```

inside this directory. For the cpu version run

```bash
uv sync --group pyg-cpu --no-group pyg
```

### PyPI

Please note that the PyPI package does not support data generation or training, just inference! If you
want to train your own model please clone the github project.

```bash
# install prerequisites
pip install torch-scatter torch-sparse torch-cluster --find-links https://data.pyg.org/whl/torch-2.4.1+cu124.html
# Install tensorframes
pip install git+https://github.com/sciai-lab/tensor_frames.git@cd1addfd3c82a47095c9961ab999dcabfab4c21d
# install mldft
pip install mldft
```

Alternatively with `uv` in one go:

```bash
uv pip install mldft torch-scatter torch-sparse torch-cluster --find-links https://data.pyg.org/whl/torch-2.4.1+cu124.html git+https://github.com/sciai-lab/tensor_frames.git@cd1addfd3c82a47095c9961ab999dcabfab4c21d
```

#### Install using Conda/Mamba/Micromamba

For conda or mamba replace `micromamba` with `conda` or `mamba` below.
If you want to create the environment with CPU support only, you can replace
`environment.yaml` with `environment_cpu.yaml`.

```bash
micromamba env create -f environment.yaml  # create mamba environment
micromamba activate mldft                  # activate environment
pip install -e .                           # install as an editable package
```

#### Install using Pip

```bash
pip install -r requirements.txt -e .       # install requirements and package
```

#### Setup using script

To actually run an OFDFT calculation using an ML model you now need to get an ML model and setup
some environment variables. To make this easier we supply a little setup script that you can call
with

```bash
mldft_setup
```

It will ask you where to place the datasets and ML models. The defaults are
`$HOME/dft_data` and `$HOME/dft_models`, and it will also export them as the respective environment variables `DFT_DATA` and `DFT_MODELS`. Then, it will offer you
to download our two models from [our repo on Hugging Face](https://huggingface.co/sciai-lab/structures25/tree/main) (note that you
require the Hugging Face Python package, which you can install with `pip install huggingface_hub`).
It will furthermore offer you to download some dataset statistics, which will be required if you want to use the `SAD` guess during density optimization with the `cli`.

##### Environment variables

Before running the code, you need to set the following two environment variables: `DFT_DATA`, the path
where the data should be stored, and `DFT_MODELS`, which is the path where the training runs,
including model checkpoints, logs, and TensorBoard files, should be stored. You can set them in your
`.bashrc` or `.zshrc`:

```bash
export DFT_DATA="/path/to/data"
export DFT_MODELS="/path/to/models"
```

## Usage

This is a general usage manual. To reproduce results from our paper see [REPLICATION_GUIDE.md](REPLICATION_GUIDE.md).

We use [hydra](https://hydra.cc/docs/intro/) to manage configurations. The main configuration files are located in `configs/`.

#### Data generation

Our datasets are available at [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.0cfxpnwcs).

1. (Optional) Create your own dataset class or use the MISC dataset and provide xyz files to set which molecules should be generated.

2. (Optional) Create a config file in `configs/datagen/dataset/`

3. Run Kohn-Sham DFT on the dataset and create `.chk` files in `$DFT_DATA/dataset/kohn_sham`:

   Example: `mldft_ks dataset=<your_dataset_config_name> n_molecules=1000 start_idx=0`

4. Based on the Kohn-Sham result, do density fitting, compute energy and gradients and save as labels for the machine learning model in `$DFT_DATA/dataset/labels`:

   Example: `mldft_labelgen dataset=<your_dataset_config_name> n_molecules=-1`

5. Split the file into train, validation, and test datasets using `mldft/utils/create_dataset_splits.py`.

   Example: `python mldft/utils/create_dataset_splits.py <dataset_name>`

6. Create a train data config in `configs/ml/data` to link to the dataset, important are dataset_name and the right setting of atom types in the dataset.

7. Transform into a basis (to reduce data loading computations during training). For `Graphformer` models, use `local_frames_global_natrep`.

   Example: `python mldft/datagen/transform_dataset.py data=<your_train_data_config_name> data/transforms=local_frames_global_natrep`

8. Compute dataset statistics, important is to compute them for the transformation and the energy target that you want to use.

   Example: `python mldft/ml/compute_dataset_statistics.py data=<your_dataset_config_name>`

Now you can start [Training](#Training)

#### Training

Training can be run with:

```bash
python mldft/ml/train.py data=<train_data_config> model=<model_config>
```

Two important settings are:

- `data/transforms`: This determines whether the data has been pre-transformed. The default is `local_frames_global_natrep`, which means that both *local frames* and *global natural reparametrization* have been applied.
- `data.target_key`: The target you are training on. The default is `kin_plus_xc`, which means you train on the total kinetic energy and exchange-correlation energy and their gradient. Alternatives are `kin_minus_apbe`, which is a delta learning approach to the kinetic energy obtained from the APBE kinetic energy functional, and `tot`, which means you are training on the total electronic energy.

#### Density Optimization

###### On a Dataset

To run density optimization on a dataset in our format, you can run the following command:

```bash
python mldft/ofdft/run_density_optimization run_path=<path_to_ml_model> \
    n_molecules=<number_of_molecules> device=<device> initialization=<initialization> num_devices=<num_devices>
```

- `path_to_model`: The path to the model relative to `DFT_MODELS`.
- `n_molecules`: The number of molecules to compute.
- `device`: The device on which the computation should run, e.g., `cuda`, `cpu`, etc.
- `initialization`: The initialization to use, either `sad`, `minao`, or `hückel`. The `sad` initialization requires appropriate dataset statistics.

By default this will run on the validation set of the dataset the model was trained on, but you can overwrite `split_file_path` to use another split file and `split` to toggle between `train`, `val` and `test` splits of the dataset.
Results are plotted in the files `density_optimization.pdf` and `density_optimization_summary.pdf`.

###### On arbitrary molecules

If you want to run the density optimization on molecules in `.xyz` files that are not part of any dataset, you
can do so with:

```bash
mldft example.xyz --model /path/to/some/model
# get all options:
mldft --help
```

`--model` needs to be the path to the model directory containing a `hparams.yaml` file as well as a `checkpoints/` directory with a `last.ckpt` checkpoint.
This will only work if the model has been trained for all atom types present in the molecule. A logfile with the same base name as the `.xyz` file and the `.log` suffix will be created.
Additionally, if you have the required dataset statistics you can use the `sad` initialization, by default `minao`.
The result will be saved in a file with `.pt` ending with the same base name as your `.xyz` file.

Alternatively, if you downloaded our models using our setup script, you can also specify our models by name:

```bash
mldft xyzfile --model  str25_qm9 # or str25_qmugs
```

## Additional information

#### Build documentation

```bash
make docs
# or to build from scratch:
make docs-clean
```

#### Template

For more details about the template, visit: https://github.com/ashleve/lightning-hydra-template

## Third-party licenses

This code adapts code from the following third party libraries:

- [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric)
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

These are distributed under the

<details>
  <summary>MIT License</summary>
  <p>
    Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

</p>
</details>

## Citation

If you use this repository in your research, please cite the following paper:

```
@article{Remme_Stable_and_Accurate_2025,
    author = {Remme, Roman and Kaczun, Tobias and Ebert, Tim and Gehrig, Christof A. and Geng, Dominik and Gerhartz, Gerrit and Ickler, Marc K. and Klockow, Manuel V. and Lippmann, Peter and Schmidt, Johannes S. and Wagner, Simon and Dreuw, Andreas and Hamprecht, Fred A.},
    doi = {10.1021/jacs.5c06219},
    journal = {Journal of the American Chemical Society},
    number = {32},
    pages = {28851--28859},
    title = {{Stable and Accurate Orbital-Free Density Functional Theory Powered by Machine Learning}},
    url = {https://doi.org/10.1021/jacs.5c06219},
    volume = {147},
    year = {2025}
}
```
