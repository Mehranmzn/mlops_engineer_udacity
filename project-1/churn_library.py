"""
Churn Library Procedure

This script performs data processing, exploratory data analysis (EDA), feature engineering, 
model training, and evaluation for churn prediction. 
Artifacts are stored in 'images' and 'models' folders.

Author: MehranMzn
Date: Nov 22, 2024
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import  classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sns.set()

# Configure logging
os.environ["QT_QPA_PLATFORM"] = "offscreen"
logging.basicConfig(
    filename="logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(asctime)-15s %(message)s",
)
logger = logging.getLogger()


def import_data(path):
    """
    Load data from a CSV file into a Pandas DataFrame.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(path)


def perform_eda(df):
    """
    Perform exploratory data analysis and save plots to the 'images/eda' folder.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        None
    """
    df["Churn"] = df["Attrition_Flag"].apply(lambda val: 0 if val == "Existing Customer" else 1)
    eda_columns = ["Churn", "Customer_Age", "Marital_Status", "Total_Trans_Ct", "Heatmap"]

    for col in eda_columns:
        plt.figure(figsize=(20, 10))
        if col in ["Churn", "Customer_Age"]:
            df[col].hist()
        elif col == "Marital_Status":
            df[col].value_counts(normalize=True).plot(kind="bar")
        elif col == "Total_Trans_Ct":
            sns.histplot(df[col], kde=True, stat="density")
        elif col == "Heatmap":
            numeric_df = df.select_dtypes(include=['number'])
            sns.heatmap(numeric_df.corr(), annot=False, cmap="Dark2_r", linewidths=2)        
        plt.savefig(f"images/eda/{col}.jpg")
        plt.close()


def encoder_helper(df, category_lst, response):
    """
    Helper function to encode categorical columns with the proportion of churn.
    Args:
        df (pd.DataFrame): The input DataFrame.
        category_lst (list): List of categorical column names.
        response (str): Name of the response column (target variable).

    Returns:
        pd.DataFrame: DataFrame with encoded categorical columns.
    """
    for category in category_lst:
        churn_rates = df.groupby(category)[response].mean()
        df[f"{category}_{response}"] = df[category].map(churn_rates)
    return df


def perform_feature_engineering(df, response):
    """
    Perform feature engineering and split data into training and testing sets.

    Args:
        df (pd.DataFrame): Input DataFrame.
        response (str): Response column name.

    Returns:
        Tuple: Training and testing datasets (x_train, x_test, y_train, y_test).
    """
    features = [
        "Customer_Age", "Dependent_count", "Months_on_book", "Total_Relationship_Count",
        "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
        "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio", "Gender_Churn", "Education_Level_Churn",
        "Marital_Status_Churn", "Income_Category_Churn", "Card_Category_Churn"
    ]
    X = df[features]
    y = df[response]
    return train_test_split(X, y, test_size=0.3, random_state=42)


def classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf):
    """
    Generate and save classification reports as images.

    Args:
        y_train, y_test: Ground truth labels for training and testing data.
        y_train_preds_lr, y_train_preds_rf: Logistic regression and random forest predictions (training).
        y_test_preds_lr, y_test_preds_rf: Logistic regression and random forest predictions (testing).

    Returns:
        None
    """
    reports = {
        "Random_Forest": (y_train, y_train_preds_rf, y_test, y_test_preds_rf),
        "Logistic_Regression": (y_train, y_train_preds_lr, y_test, y_test_preds_lr),
    }
    for model, (yt, yp_train, yt_test, yp_test) in reports.items():
        plt.figure(figsize=(6, 6))
        plt.text(0.01, 1.25, f"{model} Train", fontsize=10, fontproperties="monospace")
        plt.text(0.01, 0.05, classification_report(yt, yp_train), fontsize=10, fontproperties="monospace")
        plt.text(0.01, 0.6, f"{model} Test", fontsize=10, fontproperties="monospace")
        plt.text(0.01, 0.7, classification_report(yt_test, yp_test), fontsize=10, fontproperties="monospace")
        plt.axis("off")
        plt.savefig(f"images/results/{model}_report.jpg")
        plt.close()


def feature_importance_plot(model, x_data, output_path):
    """
    Plot and save feature importance.

    Args:
        model: Trained model with feature_importances_ attribute.
        x_data (pd.DataFrame): Input feature data.
        output_path (str): Path to save the plot.

    Returns:
        None
    """
    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(f"{output_path}/Feature_Importance.jpg")
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    """
    Train models, save artifacts, and generate evaluation metrics.

    Args:
        x_train, x_test: Training and testing feature sets.
        y_train, y_test: Training and testing target sets.

    Returns:
        None
    """
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)
    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Save ROC curve
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    RocCurveDisplay.from_estimator(lrc, x_test, y_test, ax=axis, alpha=0.8)
    RocCurveDisplay.from_estimator(cv_rfc.best_estimator_, x_test, y_test, ax=axis, alpha=0.8)
    plt.savefig("images/results/Roc_Curves.jpg")
    plt.close()



    # Save classification report
    classification_report_image(
        y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf
    )

    # Save feature importance plot
    feature_importance_plot(cv_rfc, x_test, "images/results")

    # Save models
    joblib.dump(cv_rfc.best_estimator_, "models/rfc_model.pkl")
    joblib.dump(lrc, "models/logistic_model.pkl")


if __name__ == "__main__":
    logger.info("Importing data")
    df = import_data("data/bank_data.csv")

    logger.info("Performing EDA")
    perform_eda(df)

    logger.info("Encoding categorical variables")
    df = encoder_helper(
        df,
        category_columns=["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"],
        response="Churn",
    )

    logger.info("Feature engineering and splitting data")
    x_train, x_test, y_train, y_test = perform_feature_engineering(df, response="Churn")

    logger.info("Training models")
    train_models(x_train, x_test, y_train, y_test)
