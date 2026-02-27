from carprice.pipeline.predict_pipeline import PredictionPipeline

pipeline = PredictionPipeline()

sample_input = {
    "vehicle_age": 5,
    "km_driven": 40000,
    "mileage": 18.0,
    "engine": 1197,
    "max_power": 82.0,
    "seats": 5,
    "brand": "Maruti",
    "seller_type": "Individual",
    "fuel_type": "Petrol",
    "transmission_type": "Manual"
}

print(pipeline.predict(sample_input))