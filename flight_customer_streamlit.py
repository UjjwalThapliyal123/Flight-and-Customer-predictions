import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- Utility functions ---
def load_file(filename):
    try:
        df=pd.read_csv(filename)
        return df
    except:
        st.error(f"Not found{filename}")
        st.stop()
        
def load_pipeline(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"File {filename} not found.")
        st.stop()
        
navigation = st.sidebar.radio('Select',['Flight','Customer'])        

if navigation == 'Flight':
    pipeline = load_pipeline("RandomForest_pipeline.pkl")
    data = load_file("Flight_Data.csv")
    st.title("✈️ Flight Fare Prediction")
    st.markdown("Enter your flight details below:")

    airline = st.selectbox("Airline", data["Airline"].unique())
    source = st.selectbox("Source", data["Source"].unique())
    destination = st.selectbox("Destination", data["Destination"].unique())
    additional_info = st.selectbox("Additional Info", data["Additional_Info"].unique())
    total_stops = st.slider("Total Stops", min_value=0, max_value=5)

    # --- Date Selector ---
    data["DateTime"] = pd.to_datetime(data["Year"].astype(str) + "-" + data["Month"].astype(str) + "-" + data["Date"].astype(str))
    min_date = data["DateTime"].min().date()
    max_date = data["DateTime"].max().date()

    selected_date = st.date_input("Travel Date", value=min_date, min_value=min_date, max_value=max_date)
    day = selected_date.day
    month = selected_date.month
    year = selected_date.year

    # --- Create Input DataFrame ---
    input_data = pd.DataFrame([{
        "Airline": airline,
        "Source": source,
        "Destination": destination,
        "Additional_Info": additional_info,
        "Total_Stops": total_stops,
        "Date": day,
        "Month": month,
        "Year": year
    }])
    #--- Predict ---
    if st.button("🔍 Predict Price"):
        try:
            prediction = pipeline.predict(input_data)[0]
            st.success(f"🧾 Predicted Flight Price: ₹{prediction:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            
elif navigation == 'Customer':
    pipeline = load_pipeline("GradientBoostingClassifier_pipeline.pkl")
    data = load_file("passenger_data.csv")
    st.header("📋 Enter Passenger Details")

    gender = st.selectbox('Gender', data['Gender'].unique())
    customer_type = st.selectbox('Customer Type', data['Customer Type'].unique())
    travel_type = st.selectbox('Type of Travel', data['Type of Travel'].unique())
    travel_class = st.selectbox('Class', data['Class'].unique())
    age = st.slider('Age', 18, 100, 30)
    flight_distance = st.slider('Flight Distance (km)', 100, 15000, 1000)
    departure_delay = st.slider('Departure Delay (min)', 0, 600, 10)

    # === Prepare Input ===
    input_data = pd.DataFrame([[
        gender, customer_type, travel_type, travel_class, age, flight_distance, departure_delay
    ]], columns=[
        'Gender', 'Customer Type', 'Type of Travel', 'Class', 'Age', 'Flight Distance', 'Departure Delay in Minutes'
    ])
    
    if st.button("🔍 Predict Customer Behaviour"):
        try:
            prediction = pipeline.predict(input_data)[0]
            st.success(f"🧾 Predicted Customer Behaviou: {prediction}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
