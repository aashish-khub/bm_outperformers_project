

# Forecasting S&P 500 Stock Outperformance Using Machine Learning
This repository contains the implementation of a machine learning project that forecasts the relative performance of S&P 500 stocks within a one-week horizon. By framing the problem as a binary classification task, the project predicts whether a stock’s active return (relative to the benchmark) will exceed a predefined threshold of 100 basis points (1%).

The repository includes all relevant scripts, data, and documentation to replicate the project, making it a valuable resource for those interested in the intersection of machine learning and financial markets.

 ### Project Overview
This project is a part of ORIE 5160 - Final Project at Cornell University. It explores a systematic approach to stock performance forecasting using various machine learning techniques, including Logistic Regression, Random Forest, XGBoost, K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), and a Stacking Classifier that combines the strengths of multiple models.

### Key features of the project include:

Feature Engineering: Capturing historical stock performance with lagged active returns over a 12-week period.
Exploratory Data Analysis (EDA): Understanding the data distribution and identifying patterns for prediction.
Machine Learning Models: Evaluating models on key metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
Simulated Trading Strategy: Translating predictions into portfolio decisions and comparing performance against the S&P 500 benchmark.
Repository Structure


### Repository Structure:
```yaml
.
├── data/
│   ├── Raw and processed datasets
│   ├── Example: SP500_eval_predictions_LogisticRegression.csv
│   ├── Plots:
│       ├── SP500_plot_eval_XGBoost_port_vs_bm.png
│       ├── SP500_plot_eval_Stacking_port_vs_bm.png
│
├── dslc_documentation/
│   ├── functions/
│       ├── 01_cleaning.ipynb: Notebook for data cleaning and preprocessing
│       ├── 02_eda.ipynb: Exploratory Data Analysis
│       ├── 03_prediction.ipynb: Main prediction workflow
│       ├── 03_prediction_SVC.ipynb: Support Vector Classifier model
│       ├── 03_prediction_logistic_regression.ipynb: Logistic Regression model
│       ├── 03_prediction_random_forest.ipynb: Random Forest model
│       ├── 03_prediction_xgboost.ipynb: XGBoost model
│       ├── 04_trading_strategy.ipynb: Trading strategy simulation
│
├── .gitignore: Specifies files and directories ignored by Git
├── README.md: Main project documentation
.

```

### Highlights
#### Datasets:

Data sourced via Bloomberg Terminal, covering S&P 500 constituent prices (2015–2024).
Preprocessed and cleaned datasets ensure stability and computability.
#### Machine Learning Models:

Baseline: Logistic Regression.
Advanced: Random Forest, XGBoost, KNN, and SVC.
Ensemble: Stacking Classifier combining Logistic Regression, Random Forest, and XGBoost.
#### Trading Strategy:

A simulated $1M portfolio was allocated based on model predictions.
Performance evaluation included annualized returns and comparison against the benchmark.
### Key Findings:

While ML models provide actionable insights, challenges such as imbalanced data and noisy markets persist.
The ensemble model showed improved accuracy but highlighted the complexity of real-world financial forecasting.
#### How to Use
Clone the repository:

git clone https://github.com/aashish-khub/bm_outperformers_project.git


Install dependencies:

Python 3.8+ is required.
Install necessary libraries using:
Copy code
pip install -r requirements.txt
Explore notebooks for:

Data cleaning and feature engineering: 01_cleaning.ipynb
Model training and testing: 03_prediction_xgboost.ipynb
Evaluate predictions:

Check plots and CSV outputs in the data/ folder for results.
### Future Work
Incorporate alternative data sources such as sentiment analysis or macroeconomic indicators.
Explore advanced models like LSTMs or Transformers for better temporal prediction.
Integrate transaction costs and liquidity constraints for more realistic strategy simulations.
### Contributors
Aashish Khubchandani, Cornell Tech
Abhijay Rane, Cornell Tech
