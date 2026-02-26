import os
import sys
import pandas as pd
from dataclasses import dataclass

from carprice.logger import logger
from carprice.exception import CustomException


@dataclass
class DataIngestionConfig:
    source_data_path: str = "notebook/data/cardekho_dataset.csv"
    raw_data_path: str = "artifacts/data/raw.csv"


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        logger.info("Starting data ingestion")

        try:
            # Read raw dataset
            df = pd.read_csv("notebook/data/cardekho_dataset.csv")

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.config.raw_data_path, index=False)

            logger.info(
                f"Data ingestion completed. Raw data saved at {self.config.raw_data_path}"
            )

            return self.config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)