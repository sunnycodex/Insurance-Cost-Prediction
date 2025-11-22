import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """Configuration class to store the path for the preprocessor object"""
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    """Class to handle data preprocessing and transformation"""
    
    def __init__(self):
        # Initialize transformation configuration
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Create and return a preprocessing pipeline that applies:
        - StandardScaler for numerical features
        - OneHotEncoder + StandardScaler for categorical features
        """
        try:
            # Define numerical and categorical columns for the dataset
            numerical_columns = ['age', 'bmi', 'children']
            categorical_columns = ['sex', 'smoker', 'region']
            logging.info(f'Numerical columns: {numerical_columns}')
            logging.info(f'Categorical columns: {categorical_columns}')

            # Pipeline for numerical features: apply StandardScaler
            num_pipeline = Pipeline(steps=[('standard_scaler', StandardScaler())])

            # Pipeline for categorical features: apply OneHotEncoder then StandardScaler
            cat_pipeline = Pipeline(steps=[
                ('one_hot_encoder', OneHotEncoder(sparse_output=False)),
                ('scaler', StandardScaler(with_mean=False))
            ])

            # Combine both pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Load train/test data, apply preprocessing transformations, 
        and save the preprocessor object for future use.
        Returns transformed train and test arrays with target column appended.
        """
        try:
            # Load training and testing datasets using provided paths
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Train and test datasets loaded successfully')

            # Get the preprocessor object with all transformations
            preprocessing_obj = self.get_data_transformer_object()

            # Define target column name
            target_column_name = 'charges'

            # Separate input features and target variable for training set
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target variable for test set
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
        
            # Apply preprocessing transformations to training features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            
            # Apply preprocessing transformations to test features (using fitted preprocessor)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features with target variable for training set
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            
            # Combine transformed features with target variable for test set
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the fitted preprocessor object for later use in predictions
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj)
            
            logging.info('Data transformation completed successfully')
            
            # Return transformed arrays and preprocessor path
            return (train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(e, sys)


# Entry point: Execute data transformation pipeline
if __name__ == "__main__":
    try:
        data_transformation = DataTransformation()
        # Use paths from data ingestion or specify directly
        data_transformation.initiate_data_transformation(
            'artifacts/train.csv', 
            'artifacts/test.csv'
        )
    except Exception as e:
        logging.error(f"Error occurred in data transformation: {e}")
        raise CustomException(e, sys)

