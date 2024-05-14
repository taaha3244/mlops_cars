import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model_with_metrics,evaluation_pipeline


@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self) :
        self.model_trainer_config=ModelTrainerConfig()

    

    def initiate_model_trainer(self, train_array, test_array):
        try:

            """This method deals with model training"""

            logging.info("Model training  Config has begun")
            logging.info("Splitting Training and test input data into features and labels")

            X_train,y_train,X_test,y_test=(
                    train_array[:,:-1],
                    train_array[:,-1],
                    test_array[:,:-1],
                    test_array[:,-1]
                )
            logging.info("Arrays split into features and labels")

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Support Vector Machine": SVC(random_state=42)
            }

            model_report : dict =evaluation_pipeline(models,X_train , y_train, X_test, y_test)

                    # Identify the best model based on test accuracy
            best_model_name = max(model_report, key=lambda name: model_report[name]['Accuracy'])


            best_accuracy= model_report[best_model_name]['Accuracy']

            if best_accuracy<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            
            model=models[best_model_name]
            save_model=model.fit(X_train,y_train)
            predicted=save_model.predict(X_test)
            print(X_test[0])
            print(predicted)

            model_test_accuracy = accuracy_score(y_test, predicted)
            print(model_test_accuracy)

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=save_model
            )

            return model_test_accuracy

        except Exception as e:
            raise CustomException(e,sys)


        