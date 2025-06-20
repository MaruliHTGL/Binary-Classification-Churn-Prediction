# import ml package
import joblib
import os

import streamlit as st
import numpy as np
import pandas as pd

def load_encoder(encoder_file):
    load_encoder = joblib.load(open(os.path.join(encoder_file), 'rb'))
    return load_encoder

def load_scaler(scaler_file):
    loaded_scaler = joblib.load(open(os.path.join(scaler_file), 'rb'))
    return loaded_scaler
        
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model

def run_ml_app():
    st.markdown("<h2 style = 'text-align: center;'> Input Your Customer Data </h2>", unsafe_allow_html=True)

    gender = st.radio("Gender", ["Male", "Female"])
    senior = st.radio("Senior Citizen", ["Yes", "No"])
    partner = st.radio("Have Partner", ["Yes", "No"])
    dependents = st.radio("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure", min_value=0)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multi = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    protect = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    bill = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mail check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly Charges", min_value=0.00)
    total = st.number_input("Total Charges", min_value=0.00)

    st.markdown("<h2 style = 'text-align: center;'>Your Customer Data </h2>", unsafe_allow_html=True)

    df = pd.DataFrame({
    "Gender": [gender],
    "Senior Citizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "Phone Service": [phone],
    "Multiple Lines": [multi],
    "Internet Service": [internet],
    "Online Security": [security],
    "Online Backup": [backup],
    "Device Protection": [protect],
    "Tech Support": [support],
    "Streaming TV": [tv],
    "Streaming Movies": [movies],
    "Contract": [contract],
    "Paperless Billing": [bill],
    "Payment Method": [payment],
    "Monthly Charges": [monthly],
    "Total Charges": [total]
    })

    st.dataframe(df)

    st.markdown("<h2 style = 'text-align: center;'> Prediction Result </h2>", unsafe_allow_html=True)

    result = {
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'PaperlessBilling': bill,
            'MonthlyCharges': monthly,
            'TotalCharges': total,
            'MultipleLine': multi,
            'InternetService': internet,
            'OnlineSecurity': security,
            'OnlineBackup': backup,
            'DeviceProtection': protect,
            'TechSupport': support,
            'StreamingTV': tv,
            'StreamingMovies': movies,
            'Contract': contract,
            'PaymentMethod': payment
    }

    # Map geography to one-hot encoding
    multiplelines_dict = {'No': [1, 0, 0], 
                          'No phone service': [0, 1, 0], 
                          'Yes': [0, 0, 1]       
    }

    internetservice_dict = {'DSL': [1, 0, 0], 
                            'Fiber optic': [0, 1, 0], 
                            'No': [0, 0, 1]       
    }

    onlinesecurity_dict = {'No': [1, 0, 0],
                           'No internet service': [0, 1, 0],
                           'Yes': [0, 0, 1]
    }

    onlinebackup_dict = {'No': [1, 0, 0],
                         'No internet service': [0, 1, 0],
                         'Yes': [0, 0, 1]
    }

    deviceprotection_dict = {'No': [1, 0, 0],
                             'No internet service': [0, 1, 0],
                             'Yes': [0, 0, 1]
    }

    techsupport_dict = {'No': [1, 0, 0],
                        'No internet service': [0, 1, 0],
                        'Yes': [0, 0, 1]
    }

    streamingtv_dict = {'No': [1, 0, 0],
                        'No internet service': [0, 1, 0],
                        'Yes': [0, 0, 1]
    }

    streamingmovies_dict = {'No': [1, 0, 0],
                            'No internet service': [0, 1, 0],
                            'Yes': [0, 0, 1]
    }

    contract_dict = {'Month-to-month': [1, 0, 0],
                     'One year': [0, 1, 0],
                     'Two year': [0, 0, 1]
    }

    paymentmethod_dict = {'Bank transfer (automatic)': [1, 0, 0, 0],
                          'Credit card (automatic)': [0, 1, 0, 0],
                          'Electronic check': [0, 0, 1, 0],
                          'Mail check': [0, 0, 0, 1]
    }

    input_df = pd.DataFrame([result])
    encoder = load_encoder("le.pkl")    

    for col, le in encoder.items():
        input_df[col] = le.transform(input_df[col])

    encoded_result = []

    for key, value in result.items():
        if isinstance(value, (int, float)):
            encoded_result.append(value)
        elif key == 'MultipleLine':
            encoded_result.extend(multiplelines_dict[value])
        elif key == 'InternetService':
            encoded_result.extend(internetservice_dict[value])
        elif key == 'OnlineSecurity':
            encoded_result.extend(onlinesecurity_dict[value])
        elif key == 'OnlineBackup':
            encoded_result.extend(onlinebackup_dict[value])
        elif key == 'DeviceProtection':
            encoded_result.extend(deviceprotection_dict[value])
        elif key == 'TechSupport':
            encoded_result.extend(techsupport_dict[value])
        elif key == 'StreamingTV':
            encoded_result.extend(streamingtv_dict[value])
        elif key == 'StreamingMovies':
            encoded_result.extend(streamingmovies_dict[value])
        elif key == 'Contract':
            encoded_result.extend(contract_dict[value])
        elif key == 'PaymentMethod':
            encoded_result.extend(paymentmethod_dict[value])
        elif key in encoder:
            encoded_result.append(input_df[key].values[0])
        elif key == 'SeniorCitizen':
            encoded_result.append(1 if value == 'Yes' else 0)

    single_array = np.array(encoded_result).reshape(1, -1)

    scaling = load_scaler("scaler.pkl")    
    scaling_array = scaling.transform(single_array)

    model = load_model("mnb.pkl")  
    threshold = 0.58

    probs = model.predict_proba(scaling_array)[:, 1]
    prediction = (probs > threshold).astype(int)

    if prediction == 1:
        st.warning("Churn")
        st.write("This customer will not renew their subscription. Take further action!")
    else:
        st.success("Not Churn")
        st.write("This customer will renew their subscription.")

