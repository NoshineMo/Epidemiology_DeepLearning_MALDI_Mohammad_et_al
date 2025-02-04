# Epidemiology_DeepLearning_MALDI_Mohammad_et_al
🚧👷🏽‍♀️Repository in progress🛠️🚜  
This repository contains the code and resources for the article "Exploring Deep Learning Models for Analyzing MALDI-TOF Spectra for Infectious Disease Epidemiology" by Mohammad et al. (currently being submitted). The project focuses on leveraging all deep learning techniques tested to analyze MALDI-TOF mass spectrometry data, aiming to enhance the understanding and monitoring of infectious diseases.

![Alt text](images/graphical_abstract_article.png)

## Available Files and Their Functions

This repository contains the following files, which are essential for training and evaluating various neural network architectures. Each file plays a specific role in the pipeline, from data loading to model definition and performance evaluation.

**Pipeline Notebooks:**
1. **pipeline_notebook_1**
2. **pipeline_notebook_2**
3. **pipeline_notebook_3**

These notebooks serve as the main orchestrator for training and evaluating different neural network architectures, including:
- **Convolutional Neural Networks (CNNs)**
- **Temporal Convolutional Networks (TCNs)**
- **Recurrent Neural Networks (RNNs) with Bidirectional Gated Recurrent Units (BiGRU)**
- **Echo State Networks (ESNs)**
- **2D Convolutional Neural Networks for Spectrograms (2DCNN Spectrogram)**
- **Hybrid 2D Convolutional Neural Networks with Bidirectional Gated Recurrent Units for Spectrograms (2DCNN BiGRU - Hybrid Spectrogram)**
- **2D Convolutional Neural Networks for Scalograms (2DCNN Scalogram)**
- **Hybrid 2D Convolutional Neural Networks with Bidirectional Gated Recurrent Units for Scalograms (2DCNN BiGRU - Hybrid Scalogram)**
- **Fully Connected Deep AutoEncoder (fcDAE)**
- **Deep Convolutional AutoEncoder (DCAE)**

### Associated Modules:

`libraries_utils.py` – Loads all required libraries for model training and evaluation.  
`code_methodo.py` – Defines architectures for CNN, TCN, RNN-BiGRU, and ESN, along with utilities for training neural networks.  
`code_methodo_2.py` – Defines architectures for 2DCNN spectrogram, 2DCNN BiGRU - Hybrid spectrogram, 2DCNN scalogram, 2DCNN BiGRU - Hybrid scalogram, fcDAE, and DCAE, along with additional utilities for neural network training.  
`data_loader.py` – Handles the data loading process and preprocessing steps.  
`functions_utils_for_nn.py` – Implements training loops and inference functions.  
`bootstrap_performances.py`, `regression_roc_auc.py`, `code_show_result.py` – Computes evaluation metrics for performance analysis and result visualization.  

Each file is designed to be modular, making it easier to adapt and experiment with different network architectures and datasets.  

## Contents:

- Deep learning model implementations
- Training and evaluation workflows
- Documentation and usage instructions
- Data preprocessing scripts (available upon request)
- Example datasets (available upon request)

## Goals:

- Develop robust deep learning models for MALDI-TOF spectral analysis
- Improve accuracy in detecting and characterizing infectious agents
- Provide tools for epidemiological research and public health surveillance
- Feel free to explore, contribute, and use the resources provided to advance research in infectious disease epidemiology.

