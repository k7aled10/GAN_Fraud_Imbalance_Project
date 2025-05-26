# GAN_Fraud_Imbalance_Project

## Description
This project explores the use of Generative Adversarial Networks (GANs) for balancing imbalanced datasets. Specifically, it applies GANs to credit card fraud detection. The goal is to generate synthetic data for the minority class to improve classification performance on imbalanced datasets.

## Dataset
The dataset used for this project is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. It contains transactions made by credit cards, with a focus on fraudulent transactions, which are rare and form the minority class.

### Imbalance Analysis
The dataset exhibits a significant class imbalance, with the majority of transactions being non-fraudulent. The challenge is to generate synthetic data for the fraudulent class using GANs to balance the dataset.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/GAN_Fraud_Imbalance_Project.git
````

2. Navigate to the project directory:

   ```bash
   cd GAN_Fraud_Imbalance_Project
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter notebook or Python scripts in the `src/` folder.

## GAN Implementation

This project implements both a **Vanilla GAN** and an **advanced GAN variant** (e.g., **WGAN** or **CGAN**) for generating synthetic fraud transaction data. Both GANs are trained on the minority class (fraudulent transactions) to generate new samples.

## Results

The performance of the classifiers trained on:

* The original imbalanced dataset.
* The dataset balanced using Vanilla GAN.
* The dataset balanced using an advanced GAN variant.

Evaluation metrics include Accuracy, Precision, Recall, F1-Score, AUC-ROC, and Confusion Matrix.
