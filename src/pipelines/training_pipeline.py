import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def main():
    """
    Main training pipeline that orchestrates:
    1. Data Ingestion - Load and split raw data
    2. Data Transformation - Preprocess and transform features
    3. Model Training - Train and select best model
    """
    try:
        logging.info('='*50)
        logging.info('Training Pipeline Started')
        logging.info('='*50)

        # Step 1: Data Ingestion
        # =====================================================
        logging.info('Step 1: Initiating Data Ingestion')
        obj = DataIngestion()
        logging.info('DataIngestion object created successfully')
        
        # Load raw data and split into train-test sets
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        logging.info(f'Data Ingestion completed')
        logging.info(f'Training data path: {train_data_path}')
        logging.info(f'Testing data path: {test_data_path}')
        print(f'\nTrain Data Path: {train_data_path}')
        print(f'Test Data Path: {test_data_path}\n')

        # Step 2: Data Transformation
        # =====================================================
        logging.info('Step 2: Initiating Data Transformation')
        data_transformation = DataTransformation()
        logging.info('DataTransformation object created successfully')
        
        # Apply preprocessing transformations to train and test data
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info(f'Data Transformation completed')
        logging.info(f'Training array shape: {train_arr.shape}')
        logging.info(f'Testing array shape: {test_arr.shape}')
        logging.info(f'Preprocessor saved at: {preprocessor_path}')
        print(f'Preprocessor Path: {preprocessor_path}\n')

        # Step 3: Model Training
        # =====================================================
        logging.info('Step 3: Initiating Model Training')
        model_trainer = ModelTrainer()
        logging.info('ModelTrainer object created successfully')
        
        # Train multiple models and select the best one
        best_model = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f'Model Training completed')
        logging.info(f'Best model selected and saved')

        # Pipeline completion summary
        logging.info('='*50)
        logging.info('Training Pipeline Completed Successfully!')
        logging.info('='*50)
        print('\n✓ Training Pipeline completed successfully!')
        print('✓ Best model saved in artifacts folder')
        print('✓ Ready for predictions!\n')

    except Exception as e:
        # Log error details and raise custom exception
        logging.error(f'Error occurred during training pipeline execution: {str(e)}')
        logging.error(f'Exception type: {type(e).__name__}')
        raise CustomException(e, sys)


if __name__ == '__main__':
    main()

