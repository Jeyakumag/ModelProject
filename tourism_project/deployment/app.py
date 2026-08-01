import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
# ==================================
# Streamlit UI
# ==================================

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)

# ==================================
# Load Model from Hugging Face
# ==================================

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="JeyAIML/Tourism-Project-model",
        filename="best_Tourism-Project_model_v1.joblib"
    )
    return joblib.load(model_path)

model = load_model()



st.title("✈️ MLOps - Tourism Package Purchase Prediction")

st.markdown("""
This application predicts whether a customer is likely to purchase
a tourism package based on customer demographics, travel history,
and sales interaction details.
""")

st.divider()

# ==================================
# Customer Details
# ==================================

col1, col2 = st.columns(2)

with col1:

    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    TypeofContact = st.selectbox(
        "Type of Contact",
        ["Company Invited", "Self Inquiry"]
    )

    CityTier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    Occupation = st.selectbox(
        "Occupation",
        [
            "Salaried",
            "Freelancer",
            "Small Business",
            "Large Business"
        ]
    )

    Gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    NumberOfPersonVisiting = st.number_input(
        "Number Of Persons Visiting",
        min_value=1,
        max_value=10,
        value=2
    )

    PreferredPropertyStar = st.selectbox(
        "Preferred Property Star",
        [1, 2, 3, 4, 5]
    )

    MaritalStatus = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

    NumberOfTrips = st.number_input(
        "Number Of Trips",
        min_value=0,
        max_value=50,
        value=2
    )

with col2:

    Passport = st.selectbox(
        "Passport",
        ["Yes", "No"]
    )

    OwnCar = st.selectbox(
        "Own Car",
        ["Yes", "No"]
    )

    NumberOfChildrenVisiting = st.number_input(
        "Number Of Children Visiting",
        min_value=0,
        max_value=5,
        value=0
    )

    Designation = st.selectbox(
        "Designation",
        [
            "Executive",
            "Manager",
            "Senior Manager",
            "VP"
        ]
    )

    MonthlyIncome = st.number_input(
        "Monthly Income",
        min_value=5000,
        max_value=500000,
        value=50000
    )

    PitchSatisfactionScore = st.slider(
        "Pitch Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )

    ProductPitched = st.selectbox(
        "Product Pitched",
        [
            "Basic",
            "Standard",
            "Deluxe",
            "Super Deluxe"
        ]
    )

    NumberOfFollowups = st.number_input(
        "Number Of Followups",
        min_value=0,
        max_value=20,
        value=2
    )

    DurationOfPitch = st.number_input(
        "Duration Of Pitch (Minutes)",
        min_value=1,
        max_value=120,
        value=15
    )

# ==================================
# Prepare Input Data
# ==================================

input_data = pd.DataFrame([
    {
        "Age": Age,
        "CityTier": CityTier,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "PreferredPropertyStar": PreferredPropertyStar,
        "NumberOfTrips": NumberOfTrips,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "MonthlyIncome": MonthlyIncome,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "NumberOfFollowups": NumberOfFollowups,
        "DurationOfPitch": DurationOfPitch,
        "TypeofContact": TypeofContact,
        "Occupation": Occupation,
        "Gender": Gender,
        "MaritalStatus": MaritalStatus,
        "Designation": Designation,
        "ProductPitched": ProductPitched,
        "Passport": 1 if Passport == "Yes" else 0,
        "OwnCar": 1 if OwnCar == "Yes" else 0
    }
])

# Same threshold used during training
CLASSIFICATION_THRESHOLD = 0.45

# ==================================
# Predict
# ==================================

if st.button("🔮 Predict"):

    probability = model.predict_proba(input_data)[0][1]

    prediction = (
        probability >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    st.subheader("Prediction Result")

    st.metric(
        "Purchase Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.success(
            "✅ Customer is likely to purchase the tourism package."
        )
    else:
        st.error(
            "❌ Customer is unlikely to purchase the tourism package."
        )

    st.divider()

    st.write("Input Data")

    st.dataframe(input_data)
