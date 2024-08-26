# Language Detective: Native Language Identification

## Overview

This project explores the application of machine learning (ML) and artificial intelligence (AI) techniques in the Native Language Identification (NLI) task. The goal is to classify the native language (L1) of English as a Second Language (ESL) speakers based on their English text. By leveraging neural network models and the EF-Cambridge Open Language Database (EFCAMDAT), the project aims to capture cross-linguistic transfer patterns and enhance our understanding of how a speaker's first language influences their second language use.

## Models and Approach

We implemented and tested three models of increasing complexity:

1. **Baseline Model**: A simple, non-neural network model using sentence embeddings with GloVe and an SGDClassifier. This model serves as a reference point for performance comparison.
2. **BERT-based Model**: A fine-tuned BERT model from Hugging Face with a feed-forward classification layer. This model showed significant improvement over the baseline and achieved the best performance overall.
3. **LLAMA-based Model**: A zero-shot experiment using the LLaMA 3.1 model with 4-bit quantization. Although promising, this model demonstrated some challenges, including confusion between related languages and misinterpretations, indicating the need for further fine-tuning.

## Files in the Repository

- **`BERT_per_text_classifier.ipynb`**: Contains the implementation of the BERT-based model. This notebook performs data pre-processing, model training, and evaluation. It is the best-performing model in the project.
  
- **`Baseline_model.ipynb`**: The baseline model notebook. It provides a simple approach using GloVe embeddings and a linear classifier to establish a performance benchmark.
  
- **`Bert_basic.ipynb`**: A simplified version of the BERT-based model implementation for easier understanding and experimentation.

- **`Data_loader_validation.ipynb`**: A utility notebook for data loading, cleaning, and validation. It includes functions to preprocess the EFCAMDAT dataset and prepare it for model input.

- **`Llama-3.1 8b + Unsloth.ipynb`**: Contains the implementation of the LLaMA-based model. This notebook is optimized to be run on a Google Colab instance and includes all required installations within the notebook itself.

- **`utils/`**: A folder containing helper scripts and utility functions used across the notebooks for tasks such as data preprocessing and model evaluation.

- **`README.md`**: This file, providing an overview of the project, models, and the contents of the repository.

- **`environment.yml`**: A YAML file that specifies the conda environment for running the notebooks (except for the LLaMA model). It contains all the dependencies required to run the Baseline, BERT, and data processing notebooks.

## How to Use

1. Clone the repository:
   ```
   git clone https://github.com/aelashkin/LanguageDetective.git
   ```
2. Navigate to the project directory:
   ```
   cd LanguageDetective
   ```
3. Set up the environment:
   - For **BERT-based** and **Baseline** models, create a conda environment using the `environment.yml` file:
     ```
     conda env create -f environment.yml
     conda activate py39
     ```
   - For the **LLaMA-based** model, you can run the `Llama-3.1 8b + Unsloth.ipynb` notebook directly on a Google Colab instance. The notebook contains all necessary installations and configurations.

4. Open and explore the notebooks in Jupyter to understand the models and run experiments.

5. The `BERT_per_text_classifier.ipynb` notebook is recommended for users interested in the best-performing model. It includes a complete pipeline from data preprocessing to model evaluation.
