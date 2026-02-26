from carprice.components.data_ingestion import (
    DataIngestion,
    DataIngestionConfig
)
from carprice.logger import logger


def run_training_pipeline():
    logger.info("Training pipeline started")

    ingestion_config = DataIngestionConfig(
        raw_data_path="artifacts/data/raw.csv"
    )

    ingestion = DataIngestion(ingestion_config)
    raw_data_path = ingestion.initiate_data_ingestion()

    logger.info("Training pipeline finished ingestion step")
    return raw_data_path


if __name__ == "__main__":
    run_training_pipeline()