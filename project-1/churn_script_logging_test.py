"""
Testing Module for Churn Library

This script tests the functions in churn_library.py.
Artifacts produced are stored in the 'logs' folder.

Author: MehranMzn
Date: Nov 20, 2024
"""

import os
import logging
import glob
import sys
import pytest
import joblib

from churn_library import (
    import_data,
    perform_eda,
    encoder_helper,
    perform_feature_engineering,
    train_models,
)

# Configure logging
os.environ["QT_QPA_PLATFORM"] = "offscreen"
logging.basicConfig(
    filename="logs/churn_script_logging_and_tests.log",
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s: %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger()


@pytest.fixture(scope="module")
def df_raw():
    """Fixture to load raw data."""
    try:
        data = import_data("data/bank_data.csv")
        logging.info("Raw data fixture creation: OKAY")
        return data
    except FileNotFoundError as e:
        logging.error("Raw data fixture creation: file not found error")
        raise e


@pytest.fixture(scope="module")
def df_encoded(df_raw):
    """Fixture to create encoded DataFrame."""
    try:
        encoded_data = encoder_helper(
            df_raw,
            category_lst=[
                "Gender",
                "Education_Level",
                "Marital_Status",
                "Income_Category",
                "Card_Category",
            ],
            response="Churn",
        )
        logging.info("Encoded data fixture creation: Success in encoding data")
        return encoded_data
    except KeyError as e:
        logging.error("Encoded data fixture creation: Key error in encoder helper")
        raise e


@pytest.fixture(scope="module")
def df_fe(df_encoded):
    """Fixture to perform feature engineering."""
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df_encoded, response="Churn"
        )
        logging.info("Feature engineering fixture creation: Okay")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error("Feature engineering fixture creation: Error in feature engineering")
        raise e


def test_import(df_raw):
    """Test the import data function."""
    try:
        assert df_raw.shape[0] > 0
        assert df_raw.shape[1] > 0
        logging.info("Testing import_data: OKAY")
    except AssertionError as e:
        logging.error("Testing import data: Data is missing or invalid")
        raise e


def test_eda(df_raw):
    """Test the perform eda function."""
    perform_eda(df_raw)

    for img_name in ["Churn", "Customer_Age", "Marital_Status", "Total_Trans_Ct", "Heatmap"]:
        try:
            with open(f"images/eda/{img_name}.jpg", "r"):
                logging.info(f"Testing perform eda: {img_name} image created successfully :) !")
        except FileNotFoundError as e:
            logging.error(f"Testing perform eda: {img_name} image missing")
            raise e


def test_encoder_helper(df_encoded):
    """Test the encoder helper function."""
    try:
        assert df_encoded.shape[0] > 0
        assert df_encoded.shape[1] > 0
        logging.info("Testing Encoder Helper: Dimensions are okay")

        expected_columns = [
            "Gender_Churn",
            "Education_Level_Churn",
            "Marital_Status_Churn",
            "Income_Category_Churn",
            "Card_Category_Churn",
        ]
        for col in expected_columns:
            assert col in df_encoded
        logging.info("Testing Encoder Helper: Columns are encoded successfully")
    except AssertionError as e:
        logging.error("Testing Encoder Helper: Missing or invalid encoded columns!")
        raise e


def test_perform_feature_engineering(df_fe):
    """Test the Perform Feature Engineering function."""
    try:
        x_train, x_test, y_train, y_test = df_fe
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as e:
        logging.error("Testing Perform Feature Engineering: Mismatch in data Shapes (length)")
        raise e


def test_train_models(df_fe):
    """Test the train_models function."""
    x_train, x_test, y_train, y_test = df_fe
    train_models(x_train, x_test, y_train, y_test)

    # Check model files
    try:
        joblib.load("models/rfc_model.pkl")
        joblib.load("models/logistic_model.pkl")
        logging.info("Testing train_models: Models saved successfully")
    except FileNotFoundError as e:
        logging.error("Testing train_models: Model files not found")
        raise e

    # Check generated images
    for img_name in ["Logistic_Regression_report", "Random_Forest_report", "Feature_Importance", "Roc_Curves"]:
        try:
            with open(f"images/results/{img_name}.jpg", "r"):
                logging.info(f"Testing train_models: {img_name} image created successfully")
        except FileNotFoundError as e:
            logging.error(f"Testing train_models: {img_name} image missing")
            raise e


if __name__ == "__main__":
    # Cleanup existing artifacts before testing
    for directory in ["logs", "images/eda", "images/results", "models"]:
        files = glob.glob(f"{directory}/*")
        for file in files:
            os.remove(file)

    # Run tests
    sys.exit(pytest.main(["-s"]))
