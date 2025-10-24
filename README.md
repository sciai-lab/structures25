<div align="center">

# STRUCTURES25

[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.4.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.5-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>

</div>

![figure1](https://github.com/user-attachments/assets/00abb696-95e3-4aaa-857b-2b7548d45646)

This repository contains the code used in our publication [Stable and Accurate Orbital-Free Density Functional Theory Powered by Machine Learning](https://pubs.acs.org/doi/10.1021/jacs.5c06219). Using equivariant graph neural networks, we enable Orbital-Free Density Functional Theory calculations by learning the kinetic energy functional from data.

## Quickstart

### Inference only (using PyPI)

Installation:

```bash
pip install mldft \
torch-scatter torch-sparse torch-cluster --find-links https://data.pyg.org/whl/torch-2.4.1+cu124.html \
git+https://github.com/sciai-lab/tensor_frames.git@cd1addfd3c82a47095c9961ab999dcabfab4c21d
```

Use our setup script to download models and set environment variables:

```bash
mldft_setup
```

Run inference on xyz files using our model either trained on QM9 (`str25_qm9`) or QMUGS (`str25_qmugs`):

```bash
mldft example.xyz --model str25_qm9
```

### Full Research Workflow

Installation using uv:

```bash
git clone https://github.com/sciai-lab/structures25.git
cd structures25
uv sync
```

For installation instructions without uv or cpu support visit the [Installation Guide](https://sciai-lab.github.io/structures25/installation.html). If you have built the docs locally, open `docs/build/html/installation.html`.
Now you can either go to the replication guide [REPLICATION_GUIDE.md](REPLICATION_GUIDE.md) to reproduce results from our paper or see the usage below.

## Usage

The full usage manual now lives in our documentation. Visit the [Usage Guide](https://sciai-lab.github.io/structures25/usage.html) for detailed instructions on data generation, training, and density optimisation workflows. To reproduce the results from our paper, continue to use [REPLICATION_GUIDE.md](REPLICATION_GUIDE.md). After building the docs locally, you can open `docs/build/html/usage.html`.

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
