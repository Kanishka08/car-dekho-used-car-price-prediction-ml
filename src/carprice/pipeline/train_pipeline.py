from carprice.components.data_ingestion import DataIngestion, DataIngestionConfig
from carprice.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig
)
from carprice.components.model_trainer import ModelTrainer, ModelTrainerConfig
from carprice.logger import logger


def run_training_pipeline():
    logger.info("Training pipeline started")

    # Data Ingestion
    ingestion = DataIngestion(DataIngestionConfig())
    raw_data_path = ingestion.initiate_data_ingestion()

    # Data Transformation
    transformation = DataTransformation(DataTransformationConfig())
    X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(
        raw_data_path
    )
    logger.info("Training pipeline completed transformation step")

    # Model Training
    trainer = ModelTrainer(ModelTrainerConfig())
    trainer.initiate_model_trainer(
        X_train, X_test, y_train, y_test
    )

    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    run_training_pipeline()