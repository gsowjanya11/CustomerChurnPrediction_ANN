import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as ply
import tensorflow

# Set page title and layout
st.set_page_config(page_title="Customer Data Input", layout="centered")
st.title("📊 Customer Churn Prediction")

st.sidebar.header("User Input Features")

# 1. Geography (Categorical)
geography = st.sidebar.selectbox(
    "Geography", options=["France", "Spain", "Germany"], index=0
)

# 2. Gender (Categorical)
gender = st.sidebar.radio("Gender", options=["Female", "Male"], index=0)

# 3. Age (Integer)
age = st.sidebar.number_input(
    "Age", min_value=18, max_value=100, value=34, step=1
)

# 4. Credit Score (Integer)
credit_score = st.sidebar.number_input(
    "Credit Score", min_value=0, max_value=850, value=600, step=1
)

# 5. Tenure (Integer)
tenure = st.sidebar.number_input(
    "Tenure (Years)", min_value=0, max_value=10, value=2, step=1
)

# 6. Balance (Float/Numeric)
balance = st.sidebar.number_input(
    "Balance ($)", min_value=0.0, max_value=500000.0, value=10.0, step=0.01
)

# 7. Number of Products (Integer)
num_of_products = st.sidebar.number_input(
    "Number of Products", min_value=0, max_value=4, value=1, step=1
)

# 8. Estimated Salary (Float/Numeric)
estimated_salary = st.sidebar.number_input(
    "Estimated Salary ($)",
    min_value=0.0,
    max_value=300000.0,
    value=100.0,
    step=0.01,
)

# 9. Has Credit Card (Binary -> Map to 1 or 0)
has_cr_card_bool = st.sidebar.checkbox("Has Credit Card", value=True)
has_cr_card = 1 if has_cr_card_bool else 0

# 10. Is Active Member (Binary -> Map to 1 or 0)
is_active_bool = st.sidebar.checkbox("Is Active Member", value=True)
is_active_member = 1 if is_active_bool else 0

# Construct the requested dictionary dynamically
test_data = {
    "CreditScore": credit_score,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary,
    "Geography": geography,
}


from tensorflow.keras.models import load_model
import pickle

model=load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

test_df=pd.DataFrame([test_data])
test_df['Gender']=label_encoder_gender.transform(test_df['Gender'])
geo_df=onehot_encoder_geo.transform(test_df[['Geography']]).toarray()
test_df=pd.concat([test_df,pd.DataFrame(geo_df,columns=onehot_encoder_geo.get_feature_names_out())],axis=1).drop('Geography',axis=1)
test_df=scaler.transform(test_df)
test_pred=model.predict(test_df)
pred_prob=test_pred[0][0]



# Main Panel Layout

st.write(
    "Modify the variables in the left sidebar to see the real-time dictionary payload updating below:"
)

# Display as clean JSON/Dictionary format
st.json(test_data)
if pred_prob>0.5:
    st.title( f"Customer is likely to churn with predict probabily of: "f"{round(pred_prob*100)}%")
    
else:
   st.title( f"Customer is not likely to churn with predict probabily of: "f"{round(pred_prob*100)}%")

   
# Display as a metrics grid for readability
st.subheader("Data Highlights")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Age", value=f"{test_data['Age']} yrs")
    st.metric(label="Geography", value=test_data["Geography"])
with col2:
    st.metric(label="Credit Score", value=test_data["CreditScore"])
    st.metric(label="Balance", value=f"${test_data['Balance']:,.2f}")
with col3:
    st.metric(
        label="Active Member", value="Yes" if test_data["IsActiveMember"] else "No"
    )
    st.metric(label="Products",value=test_data["NumOfProducts"])