# Predict-House-Price-
This project implements a machine learning model to predict house prices based on various features. It utilizes a Linear Regression algorithm to learn the relationship between house characteristics and their corresponding prices.

## Overview

The goal of this project is to build a regression model that can accurately estimate house prices given features. This can be valuable for individuals looking to buy or sell property, as well as for real estate professionals and market analysts.

## Key Features

* **Data Loading:** The project loads the Boston Housing dataset from a CSV file named `"Boston Dataset.csv"` using the Pandas library. The "Unnamed: 0" column is dropped during loading.
* **Data Exploration:** The first few rows of the dataset are displayed to provide a quick overview of the data structure and features.
* **Data Splitting:** The dataset is divided into training and testing sets using `train_test_split` from scikit-learn. This allows for training the model on one subset and evaluating its performance on unseen data.
* **Linear Regression Model:** A Linear Regression model from the scikit-learn library is trained on the training data.
* **Model Evaluation:** The performance of the trained model is evaluated using the Root Mean Squared Error (RMSE) metric.
* **Visualization:** A scatter plot is generated using Matplotlib to visualize the predicted house prices against the actual house prices in the test set.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* NumPy
* Matplotlib
* Seaborn
* `train_test_split` from `sklearn.model_selection`
* `LinearRegression` from `sklearn.linear_model`
* `mean_squared_error` from `sklearn.metrics`

## Getting Started

1.  **Ensure you have the required libraries installed.** You can install them using pip:

    ```bash
    pip install pandas scikit-learn numpy matplotlib seaborn
    ```

2.  **Locate the dataset file** (`Boston Dataset.csv`).
3.  **Run the Python script** (`Predict House Prices.ipynb`) to train and evaluate the model.

## Results

The project calculates and displays the Root Mean Squared Error (RMSE) on the test set. The scatter plot provides a visual comparison of predicted and actual prices.

## Further Improvements

Potential areas for future improvement include:

* Exploring other regression models.
* Performing feature engineering.
* Incorporating more features.
* Fine-tuning hyperparameters.
