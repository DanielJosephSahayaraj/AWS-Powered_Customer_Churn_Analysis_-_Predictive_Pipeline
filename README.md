# Customer Churn Analysis & Prediction

## Overview
End-to-end customer churn project combining **Power BI visualization** and **Machine Learning prediction** on AWS.

## Project Components
- **Power BI Dashboard**: Interactive analysis (see PowerBI folder)
- **ML Model**: Churn prediction using SVM and Random Forest
- **AWS Pipeline**: S3 + Glue + SageMaker ready

## How to Run ML Model Locally
```bash
pip install -r requirements.txt
python src/churn_prediction.py
