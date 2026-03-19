import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

with open('model.pkl','rb') as pkl:
    model = pickle.load(pkl)
with open('encoder.pkl','rb') as pkl1:
    encoder = pickle.load(pkl1)
with open('scaler.pkl','rb') as pkl2:
    scaler = pickle.load(pkl2)

def main():
    st.title("Restaurant Revenue Prediction Model")
    left, right = st.columns((2,2))
    location = left.selectbox("Select Location",["Downtown","Rural","Suburban"])
    cuisine = right.selectbox("Select Cuisine",["French","Indian","Italian","Japanese","Mexican"])
    rating = left.number_input("Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
    avg_mealprice = right.number_input("Average Meal Price ($)", min_value=5.0, value=25.0, step=1.0)
    seating_capacity = left.number_input("Seating Capacity", value=0, step=1)
    weekend_reserves = right.number_input("Weekend Reservations", value=0, step=1)
    weekday_reserves = left.number_input("Weekday Reservations", value=0, step=1)

    if st.button("Predict"):
        input_df = pd.DataFrame([[location,cuisine,rating,avg_mealprice,seating_capacity,weekend_reserves,weekday_reserves]],
        columns=['Location','Cuisine','Rating','Average Meal Price','Seating Capacity','Weekend Reservations','Weekday Reservations'])
        encoded_data = encoder.transform(input_df)
        scaled_data = scaler.transform(encoded_data)
        prediction = model.predict(scaled_data)
        st.metric("Predicted Revenue", f"${prediction[0]:,.2f}")
        st.snow()

if __name__ == "__main__":
    main()
