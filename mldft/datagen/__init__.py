"""Subpackage for Kohn-Sham calculations, density fitting and training data generation.

The subpackage provides all necessary functionalities to translate molecular geometry datasets into neural network training
data.

Overview
--------
The subpackage is divided into two parts: low-level modules and high-level modules. The low-level modules work on single
molecules and do all the necessary calculations. The high-level modules scale these calculations to datasets and ensure
correct saving of the data as well as parallelization.

Low-level Modules (Methods)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- :py:mod:`mldft.datagen.methods.ksdft_calculation` wraps and patches `pyscf` to save every iteration of the Kohn-Sham computation.
- :py:mod:`mldft.datagen.methods.density_fitting` fits coefficients of a new basis to the coefficients of the Kohn-Sham basis.
- :py:mod:`mldft.datagen.methods.label_generation` calculates labels for energies and gradients.
- :py:mod:`mldft.datagen.methods.save_labels_in_zarr_file` saves computed labels in a .zarr file.

High-Level Modules
^^^^^^^^^^^^^^^^^^
- :py:mod:`mldft.datagen.kohn_sham_dataset` handles Kohn-Sham calculations on a dataset.
- :py:mod:`mldft.datagen.generate_labels_dataset` handles density fitting, label generation and saving of labels on a dataset.

Datasets
^^^^^^^^
- :py:mod:`mldft.datagen.datasets.dataset` defines the interface for a dataset.
- :py:mod:`mldft.datagen.datasets.qm9` provides the QM9 dataset.
- :py:mod:`mldft.datagen.datasets.qmugs` provides the QMUGS dataset.
- :py:mod:`mldft.datagen.datasets.misc` provides a general dataset for arbitrary xyz files.


Current Timings
^^^^^^^^^^^^^^^
The current timings on large machines using 64 cores, 128 threads in parallel.

- QM9 Kohn-Sham:        8 seconds per iteration -> 300 CPU hours
- QM9 Density Fitting:  10 seconds per iteration -> 360 CPU hours
"""
