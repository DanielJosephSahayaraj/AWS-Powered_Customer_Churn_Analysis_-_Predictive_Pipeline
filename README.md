# AWS-Powered Customer Churn Analysis & Prediction Pipeline

## Overview
End-to-end customer churn solution that combines **interactive visualization** and **machine learning prediction** using AWS services and Power BI.

This project helps businesses identify customers at risk of churning and provides actionable insights through dashboards and predictive modeling.

## Project Components

### 1. Power BI Dashboard
- Interactive dashboard showing key metrics (Total Customers, Revenue, Churn Rate, etc.)
- Retention analysis by demographics, contract type, and tenure
- Filter options for deep dive analysis
- Visualizations: Churn by Tenure, Revenue trends, Customer segmentation

### 2. Machine Learning Prediction
- Built churn prediction models using **SVM** and **Random Forest**
- Data preprocessing, feature engineering, and model evaluation
- Saved models ready for deployment

### 3. AWS Data Pipeline
- Data ingestion and storage in **AWS S3**
- ETL processing with **AWS Glue**
- Data warehousing in **Amazon Redshift**
- Model training prepared for **Amazon SageMaker**
- Architecture ready for Lambda-based inference

## Tech Stack
- **Cloud**: AWS (S3, Glue, Redshift, SageMaker)
- **Visualization**: Power BI
- **ML**: Python, Scikit-learn (SVM, Random Forest)
- **Others**: Pandas, NumPy, Joblib

## Repository Structure
