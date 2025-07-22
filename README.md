# KNOWCAST: A Knowledge-augmented Framework for Urban Rail OD Prediction

KNOWCAST is a deep learning framework for predicting time-series Origin-Destination (OD) matrices in urban rail systems. It leverages a novel graph-based neural network architecture that integrates multiple types of domain knowledge from transportation science to enhance prediction accuracy. The framework is designed to model the complex spatio-temporal dependencies of passenger flow by incorporating principles from the classic four-step travel demand model: Trip Generation, Trip Distribution, and Traffic Assignment.

This repository contains the full source code for data preprocessing, model training, hyperparameter optimization, and evaluation.

-----

## Repository Structure

The repository is organized into several key directories:

  * **`/dmn_knw_gnrtr`**: (Domain Knowledge Generator) Contains all scripts for data preprocessing. This includes connecting to databases, calculating metro travel paths, generating OD matrices, and training auxiliary models for domain knowledge integration.
  * **`/lib`**: Contains core utility functions, including custom data loaders (`utils_CUROP.py`) and performance metrics (`metrics.py`).
  * **`/models`**: The core of the project, containing all PyTorch model definitions.
      * `Net_0207.py`: The main predictive model that fuses historical data with domain knowledge.
      * `OD_Net_att.py`: The underlying encoder-decoder architecture with attention.
      * `GATRUCell.py` / `GGRUCell.py`: Custom graph recurrent units used in the encoder-decoder.
  * **`/metro_components`**: Defines classes for the metro system (e.g., Station, Line, Path) and a data requester for Suzhou's metro system.
  * **`/metro_data_convertor`**: Scripts for converting and processing various forms of metro data.
  * **`/data`**: (Not included in repo, but required for execution) This directory should contain all raw data, configuration files, and processed outputs.
  * **Root Directory**: Contains main scripts for orchestration, training, and evaluation.

-----

## Workflow and Usage

The project follows a sequential workflow from data generation to model evaluation.

**1. Setup and Configuration**

  * **Database Connection**: Configure your MySQL database credentials and table names in `dmn_knw_gnrtr/processing_sql.py` and `dmn_knw_gnrtr/generating_array_OD.py`.
  * **Data Files**: Place your metro network data (e.g., `Suzhou_zhandian_no_11.xlsx`) and operational data in a directory (e.g., `/data/suzhou_03_trimmed`).
  * **Main Configuration**: The primary configuration for training is handled by a YAML file (e.g., `data/config/train_sz_dim26_units96_h4c512_250503.yaml`). Update the paths and hyperparameters in this file as needed. The path to this file is set in `config.py`.

**2. Domain Knowledge Generation**
The entire data preprocessing pipeline is orchestrated by `generating_domain_knowledge_no_DO_clean.py`. This script performs a series of steps to prepare the data and domain knowledge features required for the main model. To run the full pipeline, execute:

```bash
python generating_domain_knowledge_no_DO_clean.py
```

This script will:

  * Connect to the SQL database to fetch raw trip data and generate time-stamped OD matrices (`generating_array_OD.py`).
  * Train auxiliary models for Trip Generation and Trip Distribution (`run_PYGT_0917.py`, `fit_trip_generation_model.py`).
  * Process the data into sequence-to-sequence format (`.pkl` files) suitable for the main model.
  * Generate and save other domain knowledge features.

**3. Hyperparameter Optimization (Optional)**
You can perform Bayesian hyperparameter optimization using Optuna.

```bash
python bayes_opt.py
```

This script will run multiple training trials with different hyperparameters, find the best combination based on validation MAE, and save the results.

**4. Model Training**
To train the main `Net_0207` model, run the `train_save_history.py` script.

```bash
python train_save_history.py
```

This script will:

  * Load the configuration from the YAML file specified in `config.py`.
  * Load the preprocessed datasets generated in Step 2.
  * Initialize the `Net_0207` model and execute the training loop.
  * Log all metrics and generate plots for analysis.

**5. Evaluation**
To evaluate a trained model on the test set, use the `evaluate_button.py` script. Update the evaluation configuration file (e.g., `data/config/eval_sz_dim... .yaml`) to point to your saved model path (`save_path`). Run the script:

```bash
python evaluate_button.py
```

This will load the model, run it on the test dataset, calculate the final performance metrics, and save the raw predictions (`test_pred.npy`) and ground truth (`test_true.npy`).

-----

## File-Specific Documentation

### `dmn_knw_gnrtr/fit_trip_generation_model.py`

  * **Purpose**: Implements and fits a doubly constrained gravity model for trip distribution. It optimizes parameters to predict OD flow ($q\_v$) based on total departures ($O$), total arrivals ($D$), and impedance ($C$).
  * **Key Functions**:
      * `impedance_function(C, gamma)`: Calculates the impedance function $f(C)=(C+\\epsilon)^{-\\gamma}$.
      * `compute_flow(O, D, C, gamma, a, b)`: Computes the predicted OD flow matrix using the gravity model formula.
      * `objective_function(...)`: Calculates the Mean Squared Error (MSE) between predicted and observed flows.
      * `fit_trip_generation_model(...)`: The main function that uses the Adam optimizer to find the optimal parameters ($\\gamma, a, b$).
  * **Usage**: Called by the main data generation pipeline, taking tensors for departures, arrivals, observed flows, and an impedance matrix as input.
  * **Outputs**: The optimized parameters and a list of predicted flow matrices.

### `dmn_knw_gnrtr/generating_array_OD.py`

  * **Purpose**: Extracts trip data from a MySQL database and transforms it into structured NumPy arrays (`(X, y)` sequences) for sequence-to-sequence modeling.
  * **Key Functions**:
      * `Connect_to_SQL(...)`: Connects to MySQL, executes a query, and loads data into a Pandas DataFrame.
      * `generate_OD_DO_array(...)`: Aggregates raw trip records into time-stamped OD matrices using multithreading.
      * `generating_trip_generation_data_and_OD_dict(...)`: Transforms the time-series of OD matrices into overlapping input/target sequences and generates corresponding historical context sequences (previous day/week).
  * **Usage**: Called by the main pipeline for each dataset split (train, test, val).
  * **Outputs**: Multiple `.pkl` files containing the time-stamped OD matrices, the final input/target sequences, and historical context data.

### `dmn_knw_gnrtr/Generating_Metro_Related_data.py`

  * **Purpose**: Generates path-related domain knowledge by finding the top-K shortest paths for every OD pair using graph traversal algorithms. This information is used for traffic assignment modeling.
  * **Key Functions**:
      * `dijkstra(...)`: Standard Dijkstra's algorithm for the single shortest path.
      * `yen_ksp(...)`: Yen's K-shortest path algorithm to find k distinct shortest paths.
      * `Generating_Metro_Related_data(...)`: The main function that loads the metro network, builds the graph, and iterates through all OD pairs to find and save their paths.
  * **Usage**: Run once for a static metro network. It's called by the main data pipeline.
  * **Outputs**: A `train_dict.pkl` file containing dictionaries that map OD pairs to paths (`OD_path_dic`), metro sections to paths (`section_path_dic`), and paths to sections (`path_section_dic`).

### `dmn_knw_gnrtr/generating_OD_path_array.py`

  * **Purpose**: Creates time-varying feature arrays for each OD pair, describing the characteristics (e.g., number of stations, transfers) of the top-3 available paths at each timestamp based on the network's operational status.
  * **Key Functions**:
      * `processing_Time_DepartFreDic_item(...)`: For a single timestamp, builds an adjacency matrix of operational sections and extracts path features for each OD pair. Designed for parallel execution.
      * `batch_processing(...)`: Manages the parallel processing across multiple CPU cores.
  * **Usage**: Executed by the main data generation pipeline.
  * **Outputs**: `OD_feature_array_dic.pkl`, a dictionary mapping timestamps to NumPy arrays of shape `(num_stations, num_stations, 3, 2)` containing features for the top 3 paths for each OD pair.

### `dmn_knw_gnrtr/generating_OD_section_pssblty_sparse_array_0209.py`

  * **Purpose**: Extends the previous script by constructing a high-dimensional sparse tensor of OD-path-section relationships and performing CANDECOMP/PARAFAC (CP) tensor decomposition. This creates a compressed, low-rank representation of traffic assignment knowledge.
  * **Key Functions**:
      * `processing_Time_DepartFreDic_item(...)`: Constructs a 5D sparse tensor `(origin, destination, path, section_start, section_end)` and applies `tensorly.decomposition.parafac` to it for each timestamp.
      * `batch_processing(...)`: Manages the computationally expensive parallel execution across multiple CPUs and GPUs.
  * **Usage**: A computationally intensive part of the main data pipeline requiring significant resources.
  * **Outputs**:
      * `OD_feature_array_dic.pkl`: Same as the previous script.
      * `Date_and_time_OD_path_cp_factors_dic.pkl`: The key output, mapping timestamps to the CP decomposition factor matrices.

### `dmn_knw_gnrtr/run_PYGT_0917.py`

  * **Purpose**: Trains an auxiliary Recurrent Graph Convolutional Network (DCRNN) to predict trip generation (production) and attraction for each station. This trained model provides domain knowledge to the main `Net_0207` model.
  * **Key Functions**:
      * `run_PYGT(...)`: The main training function that loads `StaticGraphTemporalSignal` data, initializes the `RecurrentGCN` model, and runs the training loop to minimize MSE and MAPE loss.
  * **Usage**: Called by the main data pipeline to train separate models for "production" (prdc) and "attraction" (attr).
  * **Outputs**: The saved model weights (`.pth`), model hyperparameters (`.pkl`), and the test dataset (`.pkl`).

### `lib/utils_CUROP.py`

  * **Purpose**: A central utility library providing essential functions for data handling, including data loading, batching, feature scaling, and graph preparation for PyTorch Geometric models.
  * **Key Classes**:
      * `DataLoader`: A custom data loader for the project's complex data structure, supporting batching, shuffling, and lazy loading using memory-mapped files.
      * `StandardScaler` / `StandardScaler_Torch`: Classes for standardizing data in NumPy and PyTorch.
  * **Key Functions**:
      * `load_dataset(...)`: Main function to load all data from pickle files, orchestrate feature scaling, and initialize the `DataLoader`.
      * `collate_wrapper(...)`: A critical function that transforms a batch of data into the `torch_geometric.data.Batch` format expected by the GNN models.
  * **Usage**: Used extensively by the training (`train_save_history.py`) and evaluation (`evaluate_button.py`) scripts.

### `models/Net_0207.py`

  * **Purpose**: Defines the core predictive model, `Net_0207`. This novel architecture dynamically integrates multiple streams of transportation domain knowledge (trip generation, distribution, assignment) into its forward pass.
  * **Architecture**: An encoder-decoder network based on `ODNet_att`. Its innovation is a forward pass that acts as a real-time data processing pipeline.
  * **Key Components**:
      * `ODNet_att`: The underlying graph-based encoder-decoder.
      * **Domain Knowledge Layers**: Includes a `UtilityLayer`, `LogitLayer`, `SimpleAutoencoder`, and `GravityModelNetwork` to process different knowledge types.
      * **Dynamic Feature Construction**: The unconventional forward pass dynamically calculates path choice probabilities, builds an effective tensor $T\_{eff}$ for traffic assignment, runs the pre-trained trip generation model, and computes a gravity model prediction. It then compresses these features using autoencoders and concatenates them with historical data before feeding them into the `ODNet_att` module for the final prediction.
  * **Usage**: Instantiated and used within `train_save_history.py`. The inclusion of different domain knowledge components is controlled by the main YAML configuration.

### `models/OD_Net_att.py`

  * **Purpose**: Defines `ODNet_att`, the foundational encoder-decoder architecture for `Net_0207`, providing a flexible graph-based sequence-to-sequence framework with attention.
  * **Architecture**: A multi-layer recurrent encoder-decoder.
      * **Encoder**: Processes multiple input streams (main flow, long/short-term history) and uses a multi-head attention mechanism to fuse historical hidden states.
      * **Decoder**: A symmetric multi-layer GNN-GRU decoder that uses scheduled sampling during training.
  * **Key Components**:
      * `GATRUCell` or `GGRUCell`: The core recurrent units, using either Graph Attention Networks or Relational Graph Convolutional Networks.
      * `MultiheadAttention`: A standard PyTorch attention layer.
  * **Usage**: Instantiated and used exclusively by the `Net_0207` model to perform the core sequence-to-sequence prediction on the fused feature tensor.

### `train_save_history.py`

  * **Purpose**: The main script for training the `Net_0207` model. It orchestrates the entire training and validation process, saves model checkpoints, and logs performance metrics.
  * **Key Functions**:
      * `main(args)`: The primary function that manages the end-to-end training loop.
          * **Setup**: Loads configuration, sets up logging, and selects the device.
          * **Data Loading**: Uses `utils_CUROP.load_dataset` to load data.
          * **Model Initialization**: Instantiates the `Net_0207` model, loss criterion, and optimizer.
          * **Training Loop**: Iterates through epochs, performs forward/backward passes, and logs training loss.
          * **Validation**: Periodically evaluates the model on the validation and test sets.
          * **Checkpointing**: Saves the best-performing model based on validation MAE.
          * **Logging**: Saves all metrics to an Excel file and generates plots.
          * **Early Stopping**: Monitors validation loss to stop training if there is no improvement.
  * **Usage**: The main entry point for model training. Run directly from the command line:
    ```bash
    python train_save_history.py
    ```


## Acknowledgement

This repository partially builds upon the [HIAM](https://github.com/HCPLab-SYSU/HIAM) project. We sincerely appreciate their contributions to the open-source community.
