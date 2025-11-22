import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from src.utils import evaluate_model
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    """Configuration class to store the path for the trained model object"""
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    """Class to handle model training and selection of the best performing model"""
    
    def __init__(self):
        # Initialize model trainer configuration with artifact path
        self.model_trainer_config = ModelTrainerConfig()
        logging.info('ModelTrainer initialized with configuration')

    def initiate_model_trainer(self, train_array, test_array):
        """
        Train multiple regression models, evaluate their performance,
        select the best model, and save it for future predictions.
        
        Args:
            train_array: Transformed training data with features and target
            test_array: Transformed test data with features and target
        """
        try:
            logging.info('Model training initiated')
            
            # Separate features and target variable from training array
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            logging.info(f'Training set shape - Features: {X_train.shape}, Target: {y_train.shape}')
            
            # Separate features and target variable from test array
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]
            logging.info(f'Test set shape - Features: {X_test.shape}, Target: {y_test.shape}')

            # Define dictionary of regression models to train and evaluate
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'K-Neighbors Regressor': KNeighborsRegressor(),
            }
            logging.info(f'Models to evaluate: {list(models.keys())}')

            # Evaluate all models and get R2 scores on test set
            logging.info('Starting model evaluation on training and test data')
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models=models)

            # Display evaluation results for all models
            print(model_report)
            print("\n----------------------------------------------\n")
            logging.info(f'Model evaluation completed. Report: {model_report}')

            # Find the best model based on highest R2 score
            best_model_score = max(sorted(model_report.values()))
            logging.info(f'Best model R2 score: {best_model_score}')

            # Get the name of the best performing model
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            logging.info(f'Best model name: {best_model_name}')

            # Retrieve the best model object from models dictionary
            best_model = models[best_model_name]
            logging.info(f'Best model object retrieved: {best_model}')

            # Display best model information
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found - Name: {best_model_name}, R2 Score: {best_model_score}')

            # Save the best trained model to disk for later use in predictions
            logging.info(f'Saving best model to {self.model_trainer_config.trained_model_file_path}')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f'Model saved successfully at {self.model_trainer_config.trained_model_file_path}')

            # Return the best model for optional further use
            return best_model

        except Exception as e:
            # Log error and raise custom exception with detailed error information
            logging.error(f'Exception occurred during model training: {str(e)}')
            raise CustomException(e, sys)

