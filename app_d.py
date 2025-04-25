# Load saved model, encoder, and scaler
import pickle
import pandas as pd
import numpy as np
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def predict_car_price(symboling, normalized_losses, make, fuel_type, body_style,
                      drive_wheels, engine_location, width, height, engine_type,
                      engine_size, horsepower, city_mpg, highway_mpg):
    """
    Predicts the car price based on input features.

    Parameters:
    - fuel_type (str): e.g., 'gas' or 'diesel'
    - make (str): e.g., 'toyota', 'honda', etc.
    - engine_size (float)
    - horsepower (float)
    - width (float)
    - normalized_losses (float)

    Returns:
    - Predicted price (float)
    """
    # Prepare the input DataFrame
    input_data = pd.DataFrame({
        'symboling': [symboling],
        'normalized-losses': [normalized_losses],
        'make': [make],
        'fuel-type': [fuel_type],
        'body-style': [body_style],
        'drive-wheels': [drive_wheels],
        'engine-location': [engine_location],
        'width': [width],
        'height': [height],
        'engine-type': [engine_type],
        'engine-size': [engine_size],
        'horsepower': [horsepower],
        'city-mpg': [city_mpg],
        'highway-mpg': [highway_mpg]
    })

    # Define the columns to be encoded (in the order expected by the encoder)
    categorical_cols = ['make', 'fuel-type', 'body-style', 'drive-wheels', 'engine-location', 'engine-type']

    # Apply encoding to categorical features, ensuring correct order
    input_data[categorical_cols] = encoder.transform(input_data[categorical_cols])

    # Apply scaling
    input_scaled = scaler.transform(input_data)

    # Make prediction
    predicted_price = model.predict(input_scaled)[0]

    return round(predicted_price, 2)

# 2	164	audi	gas	sedan	fwd	front	66.2	54.3	ohc	109	102	24	30	13950
price = predict_car_price(

    symboling=2,
    normalized_losses=164,
    make="audi",
    fuel_type="diesel",
    body_style="sedan",
    drive_wheels="fwd",
    engine_location="front",
    width=66.2,
    height=54.3,
    engine_type="ohc",
    engine_size=109,
    horsepower=102,
    city_mpg=24,
    highway_mpg=30
)

print("Predicted Car Price: $", price)
