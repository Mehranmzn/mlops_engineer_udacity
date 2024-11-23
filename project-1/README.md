# Predict Customer Churn

This repository contains the **Predict Customer Churn** project, part of the **ML DevOps Engineer Nanodegree** program by Udacity. The project demonstrates best practices in machine learning development, including coding standards, modularity, and testing, to predict customer churn based on credit card data.

## Project Description

The goal of this project is to identify customers who are most likely to churn based on their credit card usage patterns and behaviors. The project involves creating a clean, modular Python package that adheres to PEP8 coding standards and engineering best practices. It is a production-ready version of the exploratory work done in `churn_notebook.ipynb`.

## Files and Data Description

### Main Files
- **`churn_library.py`**  
  A Python library containing functions to process data, perform exploratory data analysis (EDA), engineer features, train machine learning models, and evaluate their performance. This library identifies customers likely to churn.
  
- **`churn_script_logging_and_tests.py`**  
  A testing script that contains unit tests for the functions in `churn_library.py`. It logs `INFO` messages and errors during test execution.

### Logs
- **Logs** are generated in the `logs/` directory. They include detailed information about script execution and testing.

### Artifacts
- **Images** generated during EDA and model evaluation are stored in the `images/` folder.
- **Models** (trained machine learning models) are saved in the `models/` folder.

## Environment Setup

Follow these steps to set up the environment and run the project:

### 1. Create a Python Environment
```bash
conda create --name churn_predict python=3.8
```
conda activate churn_predict

### 2. Test the funcitonality of the library
```bash
python churn_script_logging_and_tests.py
```
or 
```bash
pytest churn_script_logging_and_tests.py
```
This will run the tests and log the results in the `logs/` directory. The logs will contain information about the test results, including any errors or failures.
