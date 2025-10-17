# Replication Guide

We describe how to reproduce the results from our paper. If you download our data, the data generation can be skipped.

## Data generation

The computationally most expensive parts are the DFT computations and the label generation. For these you can configure
`start_idx` and `n_molecules` such that you can run the computations in parallel on multiple machines.
E.g.:

```bash
machine_0: start_idx=0 n_molecules=10000
machine_1: start_idx=10000 n_molecules=10000
...
```

For most commands you can set `num_processes` and `num_threads_per_process`. By default, `num_processes = #cpu cores - 1` and `num_threads_per_process=1`. But especially
during the label generation of large molecules you can run out of RAM when handling to many molecules in parallel. In this case reduce the number of processes and increase the number of threads per process. `max_memory_per_process` can be set to limit the RAM usage but does not provide a hard limit and should not be set too small for larger molecules.

### QM9

To generate our QM9 dataset, the following commands can be used:

```bash
python mldft/datagen/kohn_sham_dataset.py preset=qm9_perturbed_fock n_molecules=-1
python mldft/datagen/generate_labels_dataset.py preset=qm9_perturbed_fock
python mldft/datagen/transform_dataset.py data=qm9_perturbed_fock
python mldft/utils/create_dataset_splits.py QM9_perturbed_fock
python mldft/ml/compute_dataset_statistics.py data=qm9_perturbed_fock
python mldft/ml/compute_dataset_statistics.py data=qm9_perturbed_fock data/transforms=no_basis_transforms
```

### QMUGSBin0_perturbed_fock

```bash
python mldft/datagen/kohn_sham_dataset.py preset=qmugs_bin0_perturbed_fock n_molecules=-1
python mldft/datagen/generate_labels_dataset.py preset=qmugs_bin0_perturbed_fock
python mldft/datagen/transform_dataset.py data=qmugs_bin0_perturbed_fock data.dataset_name="QMUGS_perturbed_fock"
python mldft/utils/create_custom_splits.py QMUGSBin0_perturbed_fock
python mldft/ml/compute_dataset_statistics.py data=qmugs_bin0_perturbed_fock
python mldft/ml/compute_dataset_statistics.py data=qmugs_bin0_perturbed_fock data/transforms=no_basis_transforms
```

### QMUGSLargeBins

```bash
python mldft/datagen/kohn_sham_dataset.py preset=qmugs_large_bins n_molecules=-1
python mldft/datagen/generate_labels_dataset.py preset=qmugs_large_bins
python mldft/utils/create_custom_splits.py QMUGSLargeBins
```

### Merged QMUGS and QM9 train set

After creating these three datasets, you can create the final split file that has a train and validation set with QM9 and QMUGSBin0 molecules and a test set that contains the large QMUGS molecules.

```bash
python mldft/utils/create_custom_splits.py QMUGSBin0QM9_perturbed_fock
```

## Model Training

### Data transformation

If you downloaded our data, you only get the labels in the untransformed basis, but to achieve good results the data has to be transformed into local frames as well as global symmetric natural reparametrization. This can be done with the following command:

```bash
python mldft/datagen/transform_dataset.py data=qm9_perturbed_fock data/transforms=local_frames_global_natrep
python mldft/datagen/transform_dataset.py data=qmugs_bin0_perturbed_fock data/transforms=local_frames_global_natrep data.dataset_name="QMUGS_perturbed_fock"
```

### Training

To reproduce our QM9 best model training, the following command can be used for QM9:

```bash
python mldft/ml/train.py experiment=str25/qm9_tf seed=100
```

and for QMUGS:

```bash
python mldft/ml/train.py experiment=str25/qmugs_hard_cutoff_hierarc_tf seed=100
```

followed by fine-tuning:

```bash
python mldft/ml/train.py experiment=str25/qmugs_hard_cutoff_hierarc_tf weight_ckpt_path="path/to/ckpt" trainer.max_epochs=30 model.optimizer.lr=1e-5 seed=292311302
```

## Density Optimization

To run density optimization with our checkpoint, put the provided model folders into `$DFT_MODELS/train/runs`.

QM9 density optimization can be run with

```bash
python mldft/ofdft/run_density_optimization.py eval=qm9 run_path='"088__from_checkpoint_009__str25\qm9_tf"' num_devices=<num_gpus_to_use>
```

We recommend to run on multiple small GPUs.
QMUGS can be run with:

```bash
python mldft/ofdft/run_density_optimization.py eval=qmugs run_path='"214__num_workers-32__qmugs_bin0_perturbed_fock__str25\qmugs_hard_cutoff_hierarc_tf__lr-1e-5__max_epochs-30__from_weight_checkpoint_110"'
```

We recommend one large GPU with at least 24 GB of memory.
