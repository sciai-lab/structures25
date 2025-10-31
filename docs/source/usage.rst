Usage Guide
===========

This guide provides a general overview of how to work with STRUCTURES25 once the project is installed. For instructions on reproducing the experiments from our paper, refer to the `REPLICATION_GUIDE.md <https://github.com/sciai-lab/structures25/blob/main/REPLICATION_GUIDE.md>`_ in the repository.

We rely on `Hydra <https://hydra.cc/docs/intro/>`_ to manage configurations. The main configuration files are located in ``configs/``.

Data Generation
---------------

Our datasets are available at `Dryad <https://datadryad.org/dataset/doi:10.5061/dryad.0cfxpnwcs>`_.

1. (Optional) Create your own dataset class or use the MISC dataset and provide XYZ files to define which molecules should be generated.
2. (Optional) Create a config file in ``configs/datagen/dataset/``.
3. Run Kohn-Sham DFT on the dataset and create ``.chk`` files in ``$DFT_DATA/dataset/kohn_sham``:

   .. code-block:: bash

      mldft_ks dataset=<your_dataset_config_name> n_molecules=1000 start_idx=0

4. Based on the Kohn-Sham results, perform density fitting, compute energy and gradients, and save them as labels for the machine learning model in ``$DFT_DATA/dataset/labels``:

   .. code-block:: bash

      mldft_labelgen dataset=<your_dataset_config_name> n_molecules=-1

5. Split the file into train, validation, and test datasets using ``mldft/utils/create_dataset_splits.py``.

   .. code-block:: bash

      python mldft/utils/create_dataset_splits.py <dataset_name>

6. Create a training data config in ``configs/ml/data`` to link to the dataset. Ensure the ``dataset_name`` and atom types are set correctly.
7. Transform the dataset into a basis (to reduce data loading computations during training). For ``Graphformer`` models, use ``local_frames_global_natrep``.

   .. code-block:: bash

      python mldft/datagen/transform_dataset.py data=<your_train_data_config_name> data/transforms=local_frames_global_natrep

8. Compute dataset statistics, making sure to do so for the transformation and target energy you plan to use.

   .. code-block:: bash

      python mldft/ml/compute_dataset_statistics.py data=<your_dataset_config_name>

.. _training:

Training
--------

Start training with:

.. code-block:: bash

   mldft_train data=<train_data_config> model=<model_config>

Key settings:

- ``data/transforms``: Selects whether the data has been pre-transformed. The default ``local_frames_global_natrep`` applies both local frames and global natural reparametrization.
- ``data.target_key``: Determines the target you train on. The default ``kin_plus_xc`` trains on the total kinetic and exchange-correlation energy (and their gradients). Alternatives include ``kin_minus_apbe`` (delta learning relative to the APBE kinetic energy functional) and ``tot`` (total electronic energy).

Density Optimization
--------------------

On a Dataset
^^^^^^^^^^^^

To run density optimization on a dataset in the project format:

.. code-block:: bash

   mldft_denop run_path=<path_to_ml_model> \
       n_molecules=<number_of_molecules> device=<device> initialization=<initialization> num_devices=<num_devices>

- ``run_path``: Path to the model relative to ``DFT_MODELS``.
- ``n_molecules``: Number of molecules to compute.
- ``device``: Target device (for example ``cuda`` or ``cpu``).
- ``initialization``: Initialization strategy: ``sad``, ``minao``, or ``hückel``. The ``sad`` option requires matching dataset statistics.

By default the command runs on the validation split of the dataset used during training. Override ``split_file_path`` to load a different split file and ``split`` to switch between the ``train``, ``val``, and ``test`` partitions. Results are written to ``density_optimization.pdf`` and ``density_optimization_summary.pdf``.

On Arbitrary Molecules
^^^^^^^^^^^^^^^^^^^^^^

To optimize densities for molecules from standalone ``.xyz`` files:

.. code-block:: bash

   mldft example.xyz --model /path/to/some/model
   # view all options
   mldft --help

``--model`` must point to a directory containing ``hparams.yaml`` and a ``checkpoints/`` directory with a ``last.ckpt`` checkpoint. Ensure the model was trained for all atom types present in the molecule. A log file with the same basename as the ``.xyz`` file and a ``.log`` suffix is created. When dataset statistics are available you can select the ``sad`` initialization; otherwise ``minao`` is used.

If you have installed the pretrained models using ``mldft_setup``, you can reference them by name:

.. code-block:: bash

   mldft xyzfile --model str25_qm9

   # or

   mldft xyzfile --model str25_qmugs

The optimization result is saved as a ``.pt`` file matching the basename of the input ``.xyz`` file.
