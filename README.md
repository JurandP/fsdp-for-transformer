# FSDP for Transformer

This repository contains transformer training experiments utilizing Fully Sharded Data Parallel (FSDP) to distribute the model across multiple devices. This work is part of a project for the **Large Scale Machine Learning** course at the **University of Warsaw**.

## Installation and Execution

### Prerequisites
To execute this script, you will need access to a cluster that meets the following requirements:
- At least **two nodes**.
- At least **two GPUs** on one of the nodes.
- A **Slurm** workload manager configured with appropriate custom options, such as `qos`, `account`, and other cluster-specific settings.

### Running the Scripts
To run any of the provided Slurm scripts, use:
```sh
sbatch <name_of_script>.sub
```

### Available Scripts
The repository includes several Slurm submission scripts for different tasks:

- **`sbatch_grid_search.sub`** – Conducts a grid search to determine the optimal learning rate while keeping other parameters constant. **Search works on 2 nodes with 1 gpu on each.**
- **`sbatch_load.sub`** – Verifies the correctness of the model loading procedure.
- **`sbatch_save.sub`** – Ensures that the model saving procedure functions correctly.
- **`submit_2gpu.sub`** – Runs the optimized training setup for a **single node with two GPUs**.
- **`submit_begin.sub`** – Executes `initial_main.py`, a version of the training script that runs without a distributed system.
- **`submit_torchrun.sub`** – Performs a sanity check for distributed training by running a small model on a limited set of tokens from the C4 dataset.

### Dataset
This project utilizes the **C4 (Colossal Clean Crawled Corpus) dataset**. Before running any scripts, ensure that:
1. The **C4 dataset** is downloaded.
2. The paths in `main.py` are correctly set to your dataset location.

### Report
After all of executed scripts I wrote a simple report, which is available here in `report.ipynb` file. 
