# Industrial Copper Modeling
## Project Overview
This project aims to develop predictive models to address two key challenges in the copper industry: accurate pricing predictions and effective lead classification. The models leverage machine learning techniques to improve decision-making processes by handling data skewness, noise, and feature scaling.

## Tech Stack
- Python
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Streamlit
- Jupyter Notebook
- Pandas
- Numpy
- Seaborn
- Matplotlib
- Plotly
- Sk-Learn

## Problem Statement
The copper industry deals with sales and pricing data, which often suffers from skewness and noise. These issues can hinder accurate manual predictions, making the process time-consuming and suboptimal. To address these challenges, this project utilizes machine learning regression models with techniques such as data normalization, feature scaling, and outlier detection. Additionally, a lead classification model is implemented to evaluate and classify leads based on their likelihood of becoming customers.

## Pricing Prediction
A machine learning regression model is developed to predict the selling price of copper. This model addresses the following:

- Skewness and noise in the data
- Data normalization and feature scaling
- Outlier detection
- Robust algorithms to improve prediction accuracy
## Lead Classification
A lead classification model is implemented to evaluate and classify leads. The model uses the STATUS variable with WON representing success and LOST representing failure. Data points with other STATUS values are removed to focus on lead classification accuracy.
