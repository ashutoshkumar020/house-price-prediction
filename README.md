# 🏠 House Price Prediction

A beginner-friendly machine learning project that predicts house prices using Python and Linear Regression.

## What this project demonstrates

- Data loading and inspection
- Missing-value checking
- Feature and target separation
- Categorical feature encoding
- Train/test split
- Model training
- Model evaluation
- Actual vs. predicted visualization

## Features

Area, bedrooms, bathrooms, stories, main road access, guest room, basement, hot water heating, air conditioning, parking, preferred area, and furnishing status.

**Target:** `price`

## Technologies

Python, Pandas, NumPy, Scikit-learn, Matplotlib

## Model

Linear Regression using a Scikit-learn preprocessing pipeline with one-hot encoding for categorical features.

## Dataset

The included `house_data.csv` is a **synthetic educational dataset** created for this portfolio project. Its columns follow the structure commonly used in housing-price regression datasets.

## How to run

```bash
pip install -r requirements.txt
python house_price_prediction.py
```

The script prints MAE, RMSE and R² and creates `actual_vs_predicted.png`.

## Author

**Ashutosh Kumar**  
B.Tech Artificial Intelligence & Machine Learning
