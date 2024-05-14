import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle

from src.exception import CustomException
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)




# Function to evaluate models using cross-validation, accuracy, and precision
def evaluate_model_with_metrics(model, X_train, y_train, X_test, y_test):
    try:
        # Cross-validation for accuracy
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"{model.__class__.__name__} CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

        # Train the model
        model.fit(X_train, y_train)

        # Predict on test set
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)

        print(f"{model.__class__.__name__} Test Accuracy: {test_accuracy:.4f}")
        print(f"{model.__class__.__name__} Test Precision: {test_precision:.4f}\n")

        return test_accuracy, test_precision
    except Exception as e:
        raise CustomException(e, sys)
    


def evaluation_pipeline(models, X_train, y_train, X_test, y_test):
    try:
        model_performance = {}

        for name, model in models.items():
            print(f"Training and evaluating {name}...")
            test_accuracy, test_precision = evaluate_model_with_metrics(model, X_train, y_train, X_test, y_test)
            model_performance[name] = {'Accuracy': test_accuracy, 'Precision': test_precision}

        return model_performance

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
