import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🍷 Wine Prediction App")
st.sidebar.header('Wine Data')
# Input sliders
Alcohol = st.sidebar.slider('Alcohol', 11.03, 14.83, 13.0)
Alcalinity_of_ash = st.sidebar.slider('Alcalinity of ash', 10.0, 30.0, 19.0)
Magnesium = st.sidebar.slider('Magnesium', 70, 162, 100)
Total_phenols = st.sidebar.slider('Total phenols', 0.98, 3.88, 2.5)
Flavanoids = st.sidebar.slider('Flavanoids', 0.38, 5.08, 2.5)
Nonflavanoid_phenols = st.sidebar.slider('Nonflavanoid phenols', 0.13, 0.66, 0.3)
Color_intensity = st.sidebar.slider('Color intensity', 1.28, 13.0, 5.0)
Hue = st.sidebar.slider('Hue', 0.48, 1.71, 1.0)
diluted_wines = st.sidebar.slider('OD280/OD315 of diluted wines', 1.27, 4.0, 2.0)

wine_classes = {
  0: "🍇 Vintage Vibes: Crafted from Grape A (Cultivar 1)",
1: "🍷 Elegant Essence: Born of Grape B (Cultivar 2)",
2: "🥂 Sparkling Signature: Blended with Grape C (Cultivar 3)"

}

# Predict button
if st.button('Predict Wine Class'):
    input_data = np.array([[Alcohol, Alcalinity_of_ash, Magnesium, Total_phenols,
                            Flavanoids, Nonflavanoid_phenols, Color_intensity,
                            Hue, diluted_wines]])
    
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    st.success(f"Prediction: {wine_classes[prediction]}")