
# Imports
import streamlit as st
import pandas as pd
import qrcode
from io import BytesIO
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# App Title & Description
st.title("Digital Advertising CPA Prediction & Scenario Testing")

st.write("""
Upload your dataset, train a Logistic Regression model, 
and test scenarios by adjusting campaign parameters interactively.
""")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:      
    df = pd.read_csv('cleaneddata_metrics.csv')

st.write("### Preview of Data")
st.write(df.head())

# Logistic Regression model steps
target = "purchase"
features = ["w_cpa", "w_ctr", "age_group", "country", "ad_type", "day_of_week"]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

numeric_features = ["w_cpa", "w_ctr"]
categorical_features = ["age_group", "country", "ad_type", "day_of_week"]

preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

clf.fit(X_train, y_train)

# Interactive Scenario Testing User Interface
    # -------------------------------
st.write("### Scenario Testing")

w_cpa = st.number_input("Cost of Acquisition ($)", min_value=1.0, value=50.0)
w_ctr = st.slider("Click Through Rate (%)", 0.0, 1.0, 0.05)

age_group = st.selectbox("Age Group", df["age_group"].unique())
country = st.selectbox("Country", df["country"].unique())
ad_type = st.selectbox("Ad Type", df["ad_type"].unique())
day_of_week = st.selectbox("Day of Week", df["day_of_week"].unique())

# Create Scenario Dataframe for features
scenario = pd.DataFrame({
        "w_cpa": [w_cpa],
        "w_ctr": [w_ctr],
        "age_group": [age_group],
        "country": [country],
        "ad_type": [ad_type],
        "day_of_week": [day_of_week]
    })

# Predict Purchase Probability
purchase_prob = clf.predict_proba(scenario)[:, 1][0]

# Expected CPA = Cost / Expected Purchases
expected_cpa = w_cpa / purchase_prob if purchase_prob > 0 else None

# Display Results
st.write("### Scenario Results")
st.metric("Predicted Purchase Probability", f"{purchase_prob:.2%}")
if expected_cpa:
    st.metric("Expected Cost per Acquisition", f"${expected_cpa:,.2f}")
else:
    st.warning("Purchase probability is zero, CPA undefined.")

# QR Code for sharing
st.write("### Share This App")
st.write("Scan the QR code below to open this app on your phone:")

# Assume local Streamlit server URL
app_url = "http://localhost:8501"

qr = qrcode.QRCode(box_size=6, border=2)
qr.add_data(app_url)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")
buf = BytesIO()
img.save(buf)
buf.seek(0)

st.image(Image.open(buf), caption="Scan to open app")