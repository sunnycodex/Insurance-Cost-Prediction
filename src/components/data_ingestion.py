import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Configuration class to store paths for raw, train, and test data
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')  # Fixed typo: 'artifat' -> 'artifacts'
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    """Class to handle data ingestion, splitting, and saving to specified paths"""
    
    def __init__(self):
        # Initialize ingestion configuration with default paths
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Load raw data, save it, and split into train (80%) and test (20%) sets.
        Returns paths to train and test datasets.
        """
        logging.info('Entered the data ingestion method')
        try:
            # Load the medical insurance dataset from CSV file
            df = pd.read_csv('notebook/data/medical_insurance.csv')
            logging.info('Read the dataset as dataframe')
            
            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
            # Save the raw unprocessed data for reference
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Raw data saved successfully')
            
            # Split dataset into training (80%) and testing (20%) sets with random_state for reproducibility
            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save training dataset to CSV file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            # Save testing dataset to CSV file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of data is completed')

            # Return paths to train and test data for downstream processing
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Raise custom exception with error details and system info
            raise CustomException(e, sys)

        
# Entry point: Create DataIngestion object and execute data ingestion pipeline
if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()