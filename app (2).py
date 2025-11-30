import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('delivery_time_model.pkl')

st.set_page_config(page_title="10-Min Delivery Predictor", page_icon="⚡")
st.title("⚡ AI-Powered Delivery Time Predictor")
st.markdown("For quick-commerce & food delivery in Indian cities")

hour = st.slider("Order Time (24-hour)", 8, 23, 19)
order_value = st.number_input("Order Value (₹)", 100, 5000, 500)
items = st.slider("Number of Items", 1, 15, 3)
rainy = st.checkbox("Is it Raining?")
zone = st.selectbox("Delivery Zone", ["Low_Traffic", "Medium_Traffic", "High_Traffic"])

zone_map = {"Low_Traffic": 0, "Medium_Traffic": 1, "High_Traffic": 2}

if st.button("Predict Delivery Time"):
    zone_enc = zone_map[zone]
    features = [[hour, order_value, items, int(rainy), zone_enc]]
    pred = model.predict(features)[0]
    
    if pred <= 10:
        st.success(f"🚀 **{pred:.1f} minutes** — On time for 10-min promise!")
    elif pred <= 15:
        st.warning(f"⏱️ **{pred:.1f} minutes** — Slight delay expected")
    else:
        st.error(f"⚠️ **{pred:.1f} minutes** — High delay risk")
    
    st.info("💡 Tip: Reduce items or avoid peak hours for faster delivery!")