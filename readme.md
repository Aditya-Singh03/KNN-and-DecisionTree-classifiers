# K-Nearest Neighbors (k-NN) and Decision Tree Classification from Scratch


This project implements and evaluates the k-NN and Decision Tree classification algorithms from scratch in Python.

> [!NOTE]
> **I have included a jupyter notebook in this repository (`knnAndDt.ipynb`) that contains the graphs and results of my models' performance.**

## Table of Contents

- [K-Nearest Neighbors (k-NN) and Decision Tree Classification from Scratch](#k-nearest-neighbors-k-nn-and-decision-tree-classification-from-scratch)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Implementation](#implementation)
  - [Evaluation](#evaluation)
  - [Results](#results)
  - [How to Use](#how-to-use)
  - [Dependencies](#dependencies)
  - [Contributing](#contributing)

## Introduction

This project serves as a practical exploration of two widely used classification algorithms:

- **k-Nearest Neighbors (k-NN):** A simple yet effective algorithm that classifies data points based on the majority class of their k nearest neighbors.
- **Decision Tree:** A versatile algorithm that creates a tree-like model to make decisions based on feature values.

The project delves into the core concepts of these algorithms, demonstrates their implementation, and evaluates their performance on various datasets.

## Implementation

The code is structured in Python and includes classes for both k-NN and Decision Tree.

- **`knnAndDt.py`:**  Contains the following core components:
    - **KNN Class:** Implements the k-NN algorithm, including training, prediction, and accuracy calculation.
    - **Decision Tree Class:** Implements the Decision Tree algorithm, including node structure, tree building, prediction, and accuracy calculation.
    - **Helper Functions:**  Provides utility functions for calculating distances, majority class, impurity measures (Gini, entropy), and information gain.
    - **Experimentation:**  Conducts experiments on the Iris and House Votes datasets to analyze the algorithms' behavior under different conditions.

## Evaluation

The project evaluates the performance of both algorithms using the following metrics:

- **Accuracy:** The proportion of correctly classified instances.
- **Errorbar Plots (k-NN):** Visualizes the average training and testing accuracy along with standard deviation for different k values.
- **Histograms (Decision Tree):** Displays the distribution of training and testing accuracies over multiple runs.

## Results

The project presents the following results:

- **k-NN:**
    - Analysis of how k value affects accuracy on the Iris dataset, both with and without normalization.
    - Discussion on the impact of normalization on the model performance.
- **Decision Tree:**
    - Comparison of the performance using entropy and Gini impurity measures on the House Votes dataset.
    - Exploration of the impact of heuristics on model complexity and accuracy.
- **Plots:** 
    - Errorbar plots (Q1.1, Q1.2, Q1.6) illustrate the effect of k in k-NN
    - Histograms (Q2.1, Q2.2, QE.1) showcase Decision Tree accuracy distributions.

## How to Use

1. ### **Clone:** 
    Clone this repository.
    ```bash
    git clone https://github.com/Aditya-Singh03/KNN-and-DecisionTree-classifiers.git
    ```
2. ### **Install:** 
   Ensure you have the required dependencies (see below).
    ```bash
    pip install numpy pandas scikit-learn matplotlib
    ```
3. ### **How to run this program?** (its very simple and concise): 
   Execute `knnAndDt.py` to run the experiments and generate results.
   **Just type**
    ```bash
    python knnAndDt.py
    ```

    in the terminal and you will be able to run both the models and create the their plots in one go. All the plots will also get saved in this very same folder. (Just make sure that when you try to run this file, you are in the same folder as the file.)
4. ### **Explore:** Examine the generated plots and analyze the findings.


## Dependencies

- Python 3.x
- NumPy
- pandas
- scikit-learn
- matplotlib

You can install the dependencies using the following command:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Contributing

Feel free to contribute to this project by:

- Suggesting improvements to the code or documentation.
- Adding experiments on new datasets.
- Exploring alternative evaluation metrics.
- Extending the implementation to other classification algorithms. 

Let me know if you'd like any modifications to the readme! 
