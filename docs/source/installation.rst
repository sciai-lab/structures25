Installation
============

Choose the installation path that matches your needs. The PyPI package is lightweight and supports inference only. Cloning the GitHub repository unlocks the full research workflow, including data generation and training.


PyPI (Inference Only)
---------------------

Use this option if you simply want to run inference with the released checkpoints.

1. Install the PyTorch Geometric wheels and tensor frames dependency:

   .. code-block:: bash

      pip install torch-scatter torch-sparse torch-cluster --find-links https://data.pyg.org/whl/torch-2.4.1+cu124.html
      pip install git+https://github.com/sciai-lab/tensor_frames.git@cd1addfd3c82a47095c9961ab999dcabfab4c21d

2. Install MLDFT from PyPI:

   .. code-block:: bash

      pip install mldft

   Alternatively, install everything in a single call using ``uv``:

   .. code-block:: bash

      uv pip install mldft torch-scatter torch-sparse torch-cluster --find-links https://data.pyg.org/whl/torch-2.4.1+cu124.html git+https://github.com/sciai-lab/tensor_frames.git@cd1addfd3c82a47095c9961ab999dcabfab4c21d


GitHub Clone (Full Workflow)
----------------------------

Clone the repository when you need training, data generation, or to develop new functionality.

.. code-block:: bash

   git clone https://github.com/sciai-lab/structures25.git
   cd structures25


Install With UV (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   uv sync

For a CPU-only environment:

.. code-block:: bash

   uv sync --group pyg-cpu --no-group pyg


Install With Conda, Mamba, or Micromamba
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replace ``micromamba`` with ``conda`` or ``mamba`` if you prefer those tools. Use ``environment_cpu.yaml`` for CPU-only installs.

.. code-block:: bash

   micromamba env create -f environment.yaml
   micromamba activate mldft
   pip install -e .


Install With Pip
^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install -r requirements.txt -e .


Post-Install Setup
------------------

Run the helper script to configure data/model directories and download pretrained assets:

.. code-block:: bash

   mldft_setup

The script prompts for storage locations (defaults: ``$HOME/dft_data`` and ``$HOME/dft_models``), exports them as ``DFT_DATA`` and ``DFT_MODELS``, and optionally fetches pretrained QM9/QMUGS checkpoints as well as the dataset statistics needed for the ``SAD`` initialization.


Environment Variables
---------------------

Ensure the core environment variables are defined before running density optimisation:

.. code-block:: bash

   export DFT_DATA="/path/to/data"
   export DFT_MODELS="/path/to/models"

``DFT_DATA`` points to your dataset directory, while ``DFT_MODELS`` stores training runs, checkpoints, and logs. Add these exports to your shell profile (for example ``~/.bashrc`` or ``~/.zshrc``) for persistence.
