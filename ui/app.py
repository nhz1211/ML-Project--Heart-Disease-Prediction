import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("../models/final_model.pkl")

# Load dataset for visualization
df = pd.read_csv("../data/cleaned_heart_binary.csv")

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# Sidebar for navigation
menu = ["Prediction", "Data Exploration"]
choice = st.sidebar.radio("Choose an option", menu)



# ===============================
# 1. Prediction Section
# ===============================
if choice == "Prediction":
    st.title("Heart Disease Prediction App")
    st.write("Provide your health data to check the risk of heart disease.")

    # Input fields for ONLY the selected features
    thal = st.selectbox("Thal (0 = normal, 1 = fixed defect, 2 = reversible defect)", [0, 1, 2])
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0 = upsloping, 1 = flat, 2 = downsloping)", [0, 1, 2])
    ca = st.number_input("Number of Major Vessels (0–4)", min_value=0, max_value=4, step=1)
    cp = st.selectbox("Chest Pain Type (0 = typical angina, 1 = atypical angina, 2 = non-anginal, 3 = asymptomatic)", [0, 1, 2, 3])
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, step=0.1)

    features = np.array([[thal, exang, slope, ca, cp, oldpeak]])

    if st.button("Predict"):
        prediction = model.predict(features)[0]
        if prediction == 1:
            st.error("⚠️ High risk of Heart Disease")
        else:
            st.success("✅ Low risk of Heart Disease")



# ===============================
# 2. Data Exploration Section
# ===============================
elif choice == "Data Exploration":
    st.title("📊 Heart Disease Data Exploration")
    st.write("Explore patterns and trends in the dataset.")

    # Target distribution
    st.subheader("Target Distribution (0 = No Disease, 1 = Disease)")
    fig, ax = plt.subplots()
    sns.countplot(x="target", data=df, ax=ax)
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Feature importance 
    st.subheader("Feature Distributions by Target")

    feature = st.selectbox("Choose a feature to explore:", df.columns[:-1])
    plot_type = st.radio("Choose plot type:", ["Histogram", "Boxplot"])

    fig, ax = plt.subplots()

    if plot_type == "Histogram":
        sns.histplot(data=df, x=feature, hue="target", kde=True, ax=ax)
    elif plot_type == "Boxplot":
        sns.boxplot(data=df, x="target", y=feature, ax=ax)

    st.pyplot(fig)


    # Random Forest Feature Importance
    st.subheader("🔎 Feature Importance (Random Forest)")
    from sklearn.ensemble import RandomForestClassifier

    X = df.drop("target", axis=1)
    y = df["target"]

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x=importances, y=importances.index, ax=ax)
    plt.title("Feature Importance (Random Forest)")
    st.pyplot(fig)