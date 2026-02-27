import sys
import pandas as pd
import numpy as np

from carprice.utils import load_object
from carprice.exception import CustomException


class PredictionPipeline:
    def __init__(self):
        try:
            self.preprocessor = load_object("artifacts/model/preprocessor.pkl")
            self.model = load_object("artifacts/model/model.pkl")
        except Exception as e:
            raise CustomException(e, sys)

    def validate_input(self, input_data: dict):
        required_columns = [
            "vehicle_age",
            "km_driven",
            "mileage",
            "engine",
            "max_power",
            "seats",
            "brand",
            "seller_type",
            "fuel_type",
            "transmission_type"
        ]

        missing_cols = set(required_columns) - set(input_data.keys())
        if missing_cols:
            raise ValueError(f"Missing input columns: {missing_cols}")

    def predict(self, input_data: dict):
        try:
            # Validate input schema
            self.validate_input(input_data)

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Apply preprocessing
            transformed_data = self.preprocessor.transform(input_df)

            # Predict price directly (NO log reverse)
            prediction = self.model.predict(transformed_data)[0]

            return round(float(prediction), 2)

        except Exception as e:
            raise CustomException(e, sys)