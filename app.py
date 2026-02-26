import streamlit as st

from carprice.pipeline.predict_pipeline import PredictionPipeline

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Price Prediction App")
st.write("Predict the selling price of a used car using a trained ML model.")

# Initialize prediction pipeline
pipeline = PredictionPipeline()

# User Inputs
st.subheader("Enter Car Details")

vehicle_age = st.number_input(
    "Vehicle Age (years)",
    min_value=0,
    max_value=30,
    value=5
)

km_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=500000,
    value=40000
)

mileage = st.number_input(
    "Mileage (km/l)",
    min_value=5.0,
    max_value=40.0,
    value=18.0
)

engine = st.number_input(
    "Engine Capacity (cc)",
    min_value=600,
    max_value=5000,
    value=1200
)

max_power = st.number_input(
    "Max Power (bhp)",
    min_value=30.0,
    max_value=500.0,
    value=82.0
)

seats = st.selectbox(
    "Number of Seats",
    options=[2, 4, 5, 6, 7]
)

brand = st.selectbox(
    "Brand",
    options=[
        "Maruti", "Hyundai", "Honda", "Tata", "Mahindra",
        "Toyota", "Ford", "Volkswagen", "BMW", "Mercedes-Benz"
    ]
)

seller_type = st.selectbox(
    "Seller Type",
    options=["Individual", "Dealer", "Trustmark Dealer"]
)

fuel_type = st.selectbox(
    "Fuel Type",
    options=["Petrol", "Diesel", "CNG", "LPG", "Electric"]
)

transmission_type = st.selectbox(
    "Transmission Type",
    options=["Manual", "Automatic"]
)


# Prediction
if st.button("Predict Price"):
    input_data = {
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "brand": brand,
        "seller_type": seller_type,
        "fuel_type": fuel_type,
        "transmission_type": transmission_type
    }

    try:
        prediction = pipeline.predict(input_data)
        st.success(f"Estimated Selling Price: ₹ {prediction:,.0f}")
    except Exception as e:
        st.error("Something went wrong while predicting.")
        st.exception(e)