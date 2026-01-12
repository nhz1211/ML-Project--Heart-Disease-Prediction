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

# Page configuration
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 60px;
        border-radius: 10px;
        border: none;
        margin-top: 20px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border: 2px solid #ff3333;
    }
    h1 {
        color: #1f1f1f;
        font-family: 'Helvetica Neue', sans-serif;
        padding-bottom: 10px;
        border-bottom: 3px solid #ff4b4b;
    }
    h2, h3 {
        color: #2c3e50;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar styling and navigation
st.sidebar.markdown("### 🩺 Navigation")
st.sidebar.markdown("---")
menu = ["🏠 Prediction", "📊 Data Exploration", "ℹ️ About"]
choice = st.sidebar.radio("Select a page:", menu)


# ===============================
# 1. Prediction Section
# ===============================
if choice == "🏠 Prediction":
    # Header
    st.markdown("<h1 style='text-align: center;'>❤️ Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; color: #666;'>Enter your health metrics to assess cardiovascular disease risk</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🩺 Cardiac Tests")
        thal = st.selectbox(
            "Thalassemia Test Result",
            options=[0, 1, 2],
            format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x],
            help="Blood disorder test result affecting oxygen transport"
        )
        
        exang = st.selectbox(
            "Exercise Induced Angina",
            options=[0, 1],
            format_func=lambda x: ["No", "Yes"][x],
            help="Chest pain during physical activity"
        )
        
        slope = st.selectbox(
            "ST Segment Slope",
            options=[0, 1, 2],
            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
            help="Slope pattern during peak exercise on ECG"
        )
    
    with col2:
        st.markdown("#### 📋 Clinical Measurements")
        ca = st.slider(
            "Major Vessels Count",
            min_value=0,
            max_value=4,
            value=0,
            help="Number of major vessels colored by fluoroscopy"
        )
        
        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"][x],
            help="Type of chest pain experienced"
        )
        
        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            help="Depression induced by exercise relative to rest"
        )

    # Prediction button with spacing
    st.markdown("---")
    
    if st.button("🔍 Analyze Risk", key="predict"):
        features = np.array([[thal, exang, slope, ca, cp, oldpeak]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Display result with better styling
        st.markdown("<br>", unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown("""
                <div style='background-color: #ffebee; padding: 30px; border-radius: 10px; border-left: 5px solid #f44336;'>
                    <h2 style='color: #c62828; margin: 0;'>⚠️ High Risk Detected</h2>
                    <p style='font-size: 18px; color: #555; margin-top: 10px;'>
                        Risk Probability: <strong>{:.1f}%</strong><br>
                        Please consult a healthcare professional for further evaluation.
                    </p>
                </div>
            """.format(probability[1] * 100), unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background-color: #e8f5e9; padding: 30px; border-radius: 10px; border-left: 5px solid #4caf50;'>
                    <h2 style='color: #2e7d32; margin: 0;'>✅ Low Risk</h2>
                    <p style='font-size: 18px; color: #555; margin-top: 10px;'>
                        Risk Probability: <strong>{:.1f}%</strong><br>
                        Continue maintaining a healthy lifestyle!
                    </p>
                </div>
            """.format(probability[1] * 100), unsafe_allow_html=True)




# ===============================
# 2. Data Exploration Section
# ===============================
elif choice == "📊 Data Exploration":
    st.markdown("<h1 style='text-align: center;'>📊 Heart Disease Data Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; color: #666;'>Explore patterns and trends in the dataset</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Dataset overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Disease Cases", df['target'].sum())
    with col3:
        st.metric("Healthy Cases", len(df) - df['target'].sum())

    st.markdown("<br>", unsafe_allow_html=True)

    # Target distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Disease Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#5a7fb8', '#d64242']
        df['target'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=colors, ax=ax)
        ax.set_ylabel('')
        ax.set_title('Disease vs Healthy')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("📊 Target Count")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="target", data=df, palette=['#5a7fb8', '#d64242'], ax=ax)
        ax.set_xticklabels(['Healthy', 'Disease'])
        ax.set_xlabel('Condition')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Correlation heatmap
    st.subheader("🔥 Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap="RdBu_r", center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Feature exploration
    st.subheader("🔍 Feature Distribution Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        feature = st.selectbox("Choose a feature to explore:", df.columns[:-1])
    with col2:
        plot_type = st.radio("Plot type:", ["Histogram", "Boxplot"])

    fig, ax = plt.subplots(figsize=(10, 5))

    if plot_type == "Histogram":
        sns.histplot(data=df, x=feature, hue="target", kde=True, palette=['#5a7fb8', '#d64242'], ax=ax)
        ax.set_xlabel(feature.replace('_', ' ').title())
    elif plot_type == "Boxplot":
        sns.boxplot(data=df, x="target", y=feature, palette=['#5a7fb8', '#d64242'], ax=ax)
        ax.set_xticklabels(['Healthy', 'Disease'])

    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Feature Importance
    st.subheader("🏆 Top Features by Importance")
    from sklearn.ensemble import RandomForestClassifier

    X = df.drop("target", axis=1)
    y = df["target"]

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_grad = plt.cm.RdBu_r(np.linspace(0.2, 0.8, len(importances)))
    sns.barplot(x=importances.values, y=importances.index, palette=colors_grad, ax=ax)
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Features')
    ax.set_title('Random Forest Feature Importance')
    st.pyplot(fig)
    plt.close()


# ===============================
# 3. About Section
# ===============================
elif choice == "ℹ️ About":
    st.markdown("<h1 style='text-align: center;'>ℹ️ About This Application</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Purpose
    This application uses machine learning to predict the risk of heart disease based on clinical measurements 
    and test results. It's designed to assist in early detection and risk assessment.
    
    ### 🤖 Model Information
    - **Algorithm**: Random Forest Classifier (Optimized)
    - **Accuracy**: ~85-88%
    - **Features**: 6 key clinical indicators
    - **Training Data**: Cleveland Heart Disease Dataset
    
    ### 📊 Key Features Used
    1. **Thalassemia Test**: Blood disorder affecting oxygen transport
    2. **Exercise Induced Angina**: Chest pain during physical activity
    3. **ST Segment Slope**: ECG pattern during peak exercise
    4. **Major Vessels Count**: Number of vessels colored by fluoroscopy
    5. **Chest Pain Type**: Classification of chest pain symptoms
    6. **ST Depression**: Exercise-induced ECG changes
    
    ### ⚠️ Disclaimer
    This tool is for educational and informational purposes only. It should **NOT** replace professional 
    medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for 
    medical concerns.
    
    ### 👥 Credits
    Developed as part of Machine Learning coursework - Major Task Project
    
    ---
    
    <p style='text-align: center; color: #999;'>Made using Streamlit & Scikit-learn</p>
    """, unsafe_allow_html=True)