# BackPAS

**BackPAS** is a framework that integrates Machine Learning into the solution process of optimization problems, using a *predict-and-search* approach for Pseudo-Boolean Optimization (PBO) instances with backbone training. It follows a three-step pipeline:

1. **Create a dataset**  
2. **Train a model**  
3. **Generate a trust region and run a solver**

---

## Repository Structure

### `src/` – Core Scripts

This folder contains the Python scripts needed to:

- **Create datasets**
- **Train machine learning models**
- **Construct trust regions**

| File | Description |
|------|-------------|
| `create_dataset_generic.py` | Creates bipartite graphs and backbone labels for each instance in a dataset. |
| `create_trust_region_generic.py` | Uses a trained ML model to generate trust regions for a dataset partition. |
| `GCN.py` | Defines neural network architectures and dataset reader classes. |
| `helper_pas.py` | Graph creation logic used by Han et al. (2023): _"A GNN-Guided Predict-and-Search Framework for MILP"_. |
| `get_bipartite_graph.py` | Builds the bipartite graph including literal nodes. |
| `train.py` | Trains a machine learning model on a given dataset. |

### `experiments_creation/` – Experiment Setup

Contains scripts to define training, validation, and testing partitions. These are used during training and trust region generation.

| File | Description |
|------|-------------|
| `utils.py` | Helper functions to build experiment `.pkl` files that define partitions. |
| `create_experiment_from_distmiplib.py` | Builds an experiment setup from a folder of instances using filename prefixes (`train_`, `val_`, `test_easy_`, etc.). |

---

## Step-by-Step Usage

### 1. Create Dataset

Prepare your dataset as follows:

- Organize your instances and their extracted backbones:

  ```
  dataset/MIS/instance/       # Contains .opb or .lp files  
  dataset/MIS/backbone/       # Contains corresponding .backbone files  
  ```

- Files must have matching base names (e.g., `example.opb` and `example.opb.backbone`).

- Run the dataset creation script:

  ```bash
  python src/create_dataset_generic.py <path_to_dataset>
  ```

- This will generate:

  ```
  <path_to_dataset>/ml_dataset/
  └── instance_name.opb.pkl   # Pickled file with the bipartite graph and backbone labels
  ```

---

### 2. Create Experiment Partitions

Experiments define how data is split into training, validation, and test sets.

- Edit `experiments_creation/create_experiment_from_distmiplib.py`:

  | Variable | Description |
  |----------|-------------|
  | `dataset` | Name of the dataset (must match the folder name under `dataset/`) |
  | `experiment_path` | Output path for the experiment setup (should be in `wkdir/<dataset>/<experiment_name>/`) |

- Run the script:

  ```bash
  python experiments_creation/create_experiment_from_distmiplib.py
  ```

- This generates:

  ```
  wkdir/<dataset>/<experiment>/partitions/
  ├── 0.pkl       # Partition configuration  
  └── readme.txt  # Partition size info  
  ```

---

### 3. Train the Model

Train the model by running:

```bash
python src/train.py <dataset_path> <partitions_path> <log_path>
```

| Argument | Description |
|----------|-------------|
| `dataset_path` | Path to the ML dataset (e.g., `dataset/MIS/ml_dataset`) |
| `partitions_path` | Path to a partition file (e.g., `wkdir/MIS/full/partitions/0.pkl`) |
| `log_path` | Output folder for training logs and model checkpoints |

Outputs:

- `metrics.log` – Training metrics (CSV format)  
- `model_best.pth` – Best performing model checkpoint  
- `model_last.pth` – Final model checkpoint  

---

### 4. Create Trust Region

Run:

```bash
python src/create_trust_region_generic.py <dataset_path> <partition_path> <log_path> <files_limit> <P> <RP>
```

| Argument | Description |
|----------|-------------|
| `dataset_path` | Path to the dataset (not the `ml_dataset` subfolder) |
| `partition_path` | Path to the partition file from step 2 |
| `log_path` | Output folder for trust region instances |
| `files_limit` | Max number of files to process (use `99999` to disable limit) |
| `P` | Percentage of variables to include in the trust region |
| `RP` | Retrieval Precision at `P` (used to compute Delta) |

Generated structure:

```
<log_path>/test/<partition_name>/
├── orig/                                 # Original instances  
└── predict_and_search_P_RP_<P>_<RP>/     # Modified instances with trust regions  
```

> **Note:** If `files_limit` is smaller than the number of partition instances, a random subset of size `files_limit` will be used. This feature was mainly for debugging and can be ignored (use `files_limit=99999`).

---

### 5. Run a Solver

After generating trust regions, you can run a solver like **Gurobi** on the modified instances for evaluation.

---

## Citation

This code is based on the GitHub repository [Predict-and-Search_MILP_method](https://github.com/sribdcn/Predict-and-Search_MILP_method) and the following paper:

> Han, Q., Yang, L., Chen, Q., Zhou, X., Zhang, D., Wang, A., ... & Luo, X. (2023).  
> *A GNN-guided Predict-and-Search Framework for Mixed-Integer Linear Programming.*  
> [arXiv:2302.05636](https://arxiv.org/abs/2302.05636)
