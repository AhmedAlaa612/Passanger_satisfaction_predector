import streamlit as st
from streamlit_star_rating import st_star_rating
import pandas as pd
from joblib import load
import numpy as np

# Load model
model = load('final.joblib')

model_features = [
    'Customer Type', 'Type of Travel', 
    'Inflight wifi service', 'Online boarding',
    'Inflight entertainment', 'On-board service',
    'Leg room service', 'Baggage handling', 
    'Checkin service', 'Inflight service', 
    'Cleanliness'
]

st.title('Passenger Satisfaction Predictor')
st.title('✈️✈️✈️✈️✈️✈️✈️✈️✈️✈️✈️')

st.write("""
fill this form to predicts whether passenger is satisfied based on their flight experience.
""")

# form container
with st.form("satisfaction_form"):
    st.subheader("Passenger Information")
    
    col1, col2 = st.columns(2)
    with col1:
        customer_type = st.selectbox('Customer Type', ['Loyal Customer', 'disloyal Customer'], key='customer_type')
    with col2:
        travel_type = st.selectbox('Type of Travel', ['Business travel', 'Personal Travel'], key='travel_type')
    
    st.divider()
    st.subheader("Service Ratings")
    
    ratings_grid = st.columns(2)
    
    with ratings_grid[0]:
        st.write("**Flight Services**")
        wifi = st_star_rating("Inflight wifi service", 5, 3, 20, emoticons=True, key='wifi')
        boarding = st_star_rating("Online boarding", 5, 3, 20, emoticons=True, key='boarding')
        entertainment = st_star_rating("Inflight entertainment", 5, 3, 20, emoticons=True, key='entertainment')
    
    with ratings_grid[1]:
        st.write("**Cabin Services**")
        onboard = st_star_rating("On-board service", 5, 3, 20, emoticons=True, key='onboard')
        legroom = st_star_rating("Leg room service", 5, 3, 20, emoticons=True, key='legroom')
        baggage = st_star_rating("Baggage handling", 5, 3, 20, emoticons=True, key='baggage')
    
    st.divider()
    
    additional_cols = st.columns(3)
    with additional_cols[0]:
        checkin = st_star_rating("Checkin service", 5, 3, 20, emoticons=True, key='checkin')
    with additional_cols[1]:
        inflight = st_star_rating("Inflight service", 5, 3, 20, emoticons=True, key='inflight')
    with additional_cols[2]:
        cleanliness = st_star_rating("Cleanliness", 5, 3, 20, emoticons=True, key='cleanliness')
    
    submitted = st.form_submit_button("Predict Satisfaction", type="primary", use_container_width=True)

if submitted:
    inputs = {
        'Customer Type': customer_type,
        'Type of Travel': travel_type,
        'Inflight wifi service': wifi,
        'Online boarding': boarding,
        'Inflight entertainment': entertainment,
        'On-board service': onboard,
        'Leg room service': legroom,
        'Baggage handling': baggage,
        'Checkin service': checkin,
        'Inflight service': inflight,
        'Cleanliness': cleanliness
    }
    input_df = pd.DataFrame([inputs])
    
    input_df['Customer Type'] = input_df['Customer Type'].map({'Loyal Customer': 1, 'disloyal Customer': 0})
    input_df['Type of Travel'] = input_df['Type of Travel'].map({'Business travel': 1, 'Personal Travel': 0})
    
    input_df = input_df[model_features]
    prediction = model.predict(input_df)
    
    # Get probability
    try:
        probability = model.predict_proba(input_df)[0][1]
    except AttributeError:
        decision = model.decision_function(input_df)
        probability = 1 / (1 + np.exp(-decision))[0]
    
    st.subheader("Prediction Results")
    
    if prediction[0] == 1:
        st.success('✅ Prediction: Satisfied')
    else:
        st.error('❌ Prediction: Neutral or Dissatisfied')
    
    st.metric("Satisfaction Probability", f"{probability:.0%}")
    st.progress(probability)
    
    # Interpretation
    st.write("""
    **Interpretation:**
    - Above 70%: Very likely satisfied
    - 40-70%: Neutral experience
    - Below 40%: Likely dissatisfied
    """)