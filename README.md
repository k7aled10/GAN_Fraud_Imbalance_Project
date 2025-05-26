# GAN_Fraud_Imbalance_Project



## Overview

This project utilizes Generative Adversarial Networks (GAN) to balance the class distribution of a credit card fraud detection dataset. The goal is to generate synthetic fraud data (minority class) to augment the dataset, which helps in training more balanced machine learning models.

### Key Features
- Uses a GAN to generate synthetic fraud samples for an imbalanced dataset.
- Evaluates machine learning models trained on original and augmented datasets.
- Implements a **Vanilla GAN** and **WGAN (Wasserstein GAN)** for data augmentation.
- Models trained include **Random Forest Classifier** to evaluate the impact of synthetic data on fraud detection.

## Requirements

- Python 3.x
- `pandas`
- `matplotlib`
- `seaborn`
- `torch` (PyTorch)
- `scikit-learn`

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/GAN_Fraud_Imbalance_Project.git
cd GAN_Fraud_Imbalance_Project
````

### Step 2: Install dependencies

Create a virtual environment and install the required packages:

```bash
pip install -r requirements.txt
```

## How to Run the Code

1. **Load and explore the dataset:**

The dataset is preloaded in the code and will be loaded automatically on script execution. It will show the class distribution (fraud vs non-fraud).

2. **Generate Synthetic Fraud Data with GAN:**

The GAN model will generate synthetic fraud samples, which will be combined with the original non-fraud data.

3. **Train Classifiers on Both Original and Augmented Data:**

The classifiers will be trained and evaluated on:

* Original imbalanced dataset
* Augmented dataset with synthetic fraud data

4. **Run the main script:**

```bash
python main.py
```

The script will output the confusion matrix and classification report for both the **original imbalanced dataset** and the **augmented dataset** with synthetic data.

## Results

### Expected Output:

* **Confusion Matrix** and **Classification Report** for:

  * **Original dataset**
  * **Augmented dataset** using Vanilla GAN and WGAN

The output will show improved performance in terms of **recall** and **F1-score** for the fraud class when the model is trained on the augmented dataset.


