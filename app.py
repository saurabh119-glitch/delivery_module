import streamlit as st
import joblib
import json
import numpy as np

# Load model and metrics
model = joblib.load('delivery_time_model.pkl')
with open('model_metrics.json', 'r') as f:
    metrics = json.load(f)

# Page config
st.set_page_config(page_title="AI Delivery Time Predictor", page_icon="⚡")
st.title("⚡ AI Delivery Time Predictor for Quick Commerce")
st.markdown("Built for Indian food & grocery delivery startups")

# Sidebar: Model Transparency
with st.sidebar:
    st.header("🔍 About This Model")
    st.write(f"**Avg Error:** {metrics['MAE_minutes']} minutes")
    st.write(f"**Accuracy (R²):** {metrics['R2_score'] * 100:.0f}%")
    st.markdown("""
    - Trained on 1,200+ synthetic orders
    - Uses: time, order value, items, rain, traffic
    - Updated: Nov 2025
    """)
    st.info("💡 Tip: Lower error = more reliable predictions")

# Main inputs
col1, col2 = st.columns(2)
with col1:
    hour = st.slider("Order Time (24-hour)", 8, 23, 19)
    order_value = st.number_input("Order Value (₹)", 100, 5000, 500)
with col2:
    items = st.slider("Number of Items", 1, 15, 3)
    rainy = st.checkbox("Raining?")
    zone = st.selectbox("Traffic Zone", ["Low", "Medium", "High"])

# Encode zone
zone_map = {"Low": 0, "Medium": 1, "High": 2}
zone_enc = zone_map[zone]

# Predict
if st.button("🎯 Predict Delivery Time"):
    features = [[hour, order_value, items, int(rainy), zone_enc]]
    pred = model.predict(features)[0]
    
    # Visual feedback
    if pred <= 10:
        st.success(f"🚀 **{pred:.1f} minutes** — On time for 10-min promise!")
    elif pred <= 15:
        st.warning(f"⏱️ **{pred:.1f} minutes** — Slight delay expected")
    else:
        st.error(f"⚠️ **{pred:.1f} minutes** — High delay risk")
    
    # Business insight
    st.info(f"💡 This prediction has ~±{metrics['MAE_minutes']} min margin of error.")

# Footer
st.markdown("---")
st.caption("Made with ❤️ for Indian quick-commerce | [GitHub](https://github.com/yourname/delivery-time-predictor)")