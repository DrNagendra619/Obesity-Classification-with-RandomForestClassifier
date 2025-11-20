# Obesity-Classification-with-RandomForestClassifier
Obesity Classification with RandomForestClassifier
# Obesity Classification using RandomForestClassifier 🌲📊

## Overview

This repository contains a Jupyter Notebook dedicated to classifying human obesity levels using the **RandomForestClassifier** algorithm. The notebook covers a complete machine learning workflow, from initial data loading and exploration to final model evaluation.

The goal of this project is to build a robust classification model that can accurately predict an individual's obesity category based on various anthropometric and behavioral features present in the dataset.

---

## Repository Files

| File Name | Description |
| :--- | :--- |
| `Obesity_Classification_with_RandomForestClassifier.ipynb` | The main Jupyter notebook detailing the entire machine learning process, focusing specifically on the **Random Forest** model for classification. |
| `[DATASET_NAME].csv` | *Placeholder for the required dataset file (containing the features and obesity class labels).* |

---

## Technical Stack

The entire project is developed using Python within a Jupyter environment, leveraging the following libraries:

* **Data Handling:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (specifically `RandomForestClassifier`, splitting, and evaluation metrics)
* **Visualization (EDA):** `matplotlib`, `seaborn`
* **Environment:** Jupyter Notebook

---

## Methodology and Key Steps

### 1. Data Preparation and EDA

The initial steps involve preparing the dataset for the model:

* **Data Loading & Cleaning:** Loading the dataset and handling missing values, if any.
* **Feature Engineering:** Converting categorical data (like gender, family history, eating habits) into a numeric format suitable for the Random Forest model (e.g., using one-hot encoding or label encoding).
* **Exploratory Data Analysis (EDA):** Visualizing feature distributions and correlations to gain insights into the factors most strongly associated with obesity classes.

### 2. Model Training and Evaluation

The core of the project involves training and assessing the **RandomForestClassifier**:

* **Model:** **RandomForestClassifier** (an ensemble method highly effective for classification tasks).
* **Training:** The data is split (e.g., 80/20) and the model is trained on the training subset.
* **Evaluation:** Performance is measured on the unseen test set using metrics appropriate for multi-class classification:
    * **Accuracy Score**
    * **Confusion Matrix**
    * **Classification Report (Precision, Recall, F1-Score)**

**Conclusion:**
The notebook concludes by highlighting the overall accuracy and the specific performance (precision/recall) of the **RandomForestClassifier** across all identified obesity classes.

---

## Setup and Usage

To run this analysis locally, ensure you have Python installed and follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd [Your Repository Name]
    ```

2.  **Ensure the Data is Present:**
    Place your raw data file (`[DATASET_NAME].csv`) in the repository's root directory.

3.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    ```

4.  **Launch Jupyter:**
    ```bash
    jupyter notebook
    ```
    Open the `Obesity_Classification_with_RandomForestClassifier.ipynb` file to execute the cells and replicate the entire modeling process.
