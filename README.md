<!--
-*- coding: utf-8 -*-

 Author: Jantine Broek <jantine.broek@simmons-simmons.com>
 License: MIT
-->

# Transformer Proteomics


## Table of Contents
1. [General Info](#general-info)
2. [Model and Data](#model-and-data)
3. [Build and Run](#build-and-run)
4. [License](#license)



<a name="general-info"></a>
## General Info

This repository contains a project for predictive modelling of proteomics data using a Transformer-based architecture. The goal is to predict proteomics data using transcriptomics data as input. The project compares a baseline MLP model with Transformer models to evaluate their performance and suitability for this task.

The time reserved for this project was 2 weeks, although I had mainly time during the weekends and 2 evenings during the week.



<a name="model-and-data"></a>
## Model and Data

### Task Type and Model Decision
This task is a regression task, where one type of omics data is predicted using others. Both VAEs and Transformers are suitable choices: VAEs excel with noisy data and handle missing values well, while Transformers are powerful for capturing complex interactions between features through their self-attention mechanism.

I selected a Transformer model because the multi-head attention can learn different types of relationships between omics data features and capturing various correlations that might exist. The self-attention mechanism allows the model to identify important feature interactions, which I think are important in multi-omics data integration. While VAEs would also be suitable, particularly for handling noise and missing values, the potential complex interactions between omics data features made Transformers my preferred choice.

### Data
The aim is to use multiple input omics datasets to predict proteomics data. Input omics data was selected by iteratively testing combinations to determine which best predicted proteomics. This indicated that transcriptomics data only as the optimal input combination for proteomics prediction.



<a name="build-and-run"></a>
## Build and Run

### How to Build

1. Ensure you have Python 3.6 or higher installed.
2. Clone the repository:

```bash
git clone https://github.com/Jan10e/transformer_proteomics.git
cd transformer_proteomics
```

2. Create a (conda) environment. For example:

```bash
conda create -n transformer_proteomics python=3.11
conda activate transformer_proteomics
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### How to Run

1. Run the Notebook
```bash
EMBL_transformer.ipynb
```

---


