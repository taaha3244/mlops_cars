import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def data_preprocessor_method(self):
        """This function is responsible for data transformation"""

        try:
            # Categorical Columns
            categorical_cols = ['Person_Capacity', 'Size_of_Luggage']

            # Ordinal Columns
            ordinal_cols = ['Buying_Price', 'Maintenance_Price', 'No_of_Doors', 'Safety']
            
            # Define the ordering for ordinal encoding
            ordinal_categories = [
                ['low', 'med', 'high', 'vhigh'],  # Buying_Price
                ['low', 'med', 'high', 'vhigh'],  # Maintenance_Price
                ['2', '3', '4', '5more'],         # No_of_Doors
                ['low', 'med', 'high']            # Safety
            ]

            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info(f"Ordinal columns: {ordinal_cols}")

            # Pipeline for categorical data
            cat_pipeline = Pipeline(
                steps=[("one_hot_encoder", OneHotEncoder())] 
            )
            
            # Pipeline for ordinal data
            ordinal_pipeline = Pipeline(
                steps=[('ordinal', OrdinalEncoder(categories=ordinal_categories)),
                       ('scaler', StandardScaler())]
            )
            
            # Preprocessing pipelines for both categorical and ordinal columns
            preprocessor = ColumnTransformer(
                [
                    ("cat_pipeline", cat_pipeline, categorical_cols),
                    ("ordinal_pipeline", ordinal_pipeline, ordinal_cols)
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Data Transformation pipeline started")

            preprocessing_obj = self.data_preprocessor_method()

            # Separate features (X) and target (y) for both training and testing sets
            X_train = train_df.drop(columns=['Car_Acceptability'])
            y_train = train_df['Car_Acceptability']

            X_test = test_df.drop(columns=['Car_Acceptability'])
            y_test = test_df['Car_Acceptability']
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            X_train = preprocessing_obj.fit_transform(X_train)
            X_test = preprocessing_obj.transform(X_test)

            label_encoder = LabelEncoder()

            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.fit_transform(y_test)


            # Combine the transformed feature data and encoded labels
            train_arr = np.column_stack((X_train, y_train))
            test_arr = np.column_stack((X_test, y_test))



            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(f"Saved preprocessing object.")

            print(train_arr[0])
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


