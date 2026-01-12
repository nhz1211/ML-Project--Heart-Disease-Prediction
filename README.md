# ❤️ Heart Disease Prediction

A comprehensive machine learning project for predicting heart disease risk using advanced classification algorithms, feature engineering, and an interactive web application.

## 📋 Project Overview

This project implements end-to-end machine learning pipeline to predict the presence of heart disease in patients based on clinical measurements. The system includes data preprocessing, feature selection, dimensionality reduction, supervised and unsupervised learning, hyperparameter optimization, and a production-ready web interface.

## 🎯 Key Features

- ✅ **Data Preprocessing**: Cleaning, normalization, and binary/multiclass target preparation
- ✅ **Feature Selection**: Statistical methods (Chi-Square, RFE) and Random Forest importance
- ✅ **PCA Analysis**: Dimensionality reduction and variance analysis
- ✅ **Supervised Learning**: Multiple algorithms compared (Logistic Regression, Decision Tree, Random Forest, SVM)
- ✅ **Unsupervised Learning**: K-Means and Hierarchical clustering analysis
- ✅ **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV optimization
- ✅ **Model Export**: Production-ready pipeline with StandardScaler + Random Forest
- ✅ **Interactive Web App**: Streamlit-based UI for real-time predictions

## 📊 Dataset

**Cleveland Heart Disease Dataset** containing 303 patient records with 14 attributes:

- **age**: Age in years
- **sex**: Sex (1 = male, 0 = female)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting ECG results (0-2)
- **thalch**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise
- **slope**: Slope of peak exercise ST segment (0-2)
- **ca**: Number of major vessels colored by fluoroscopy (0-4)
- **thal**: Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)
- **target**: Diagnosis (1 = disease, 0 = healthy)

## 📁 Project Structure

```
ML-Project--Heart-Disease-Prediction/
├── README.md
├── data/
│   ├── heart_disease.csv              # Original dataset
│   ├── cleaned_heart_binary.csv       # Preprocessed binary classification
│   ├── cleaned_heart_multiclass.csv   # Preprocessed multiclass
│   ├── heart_pca.csv                  # PCA-transformed features
│   └── heart_selected_features.csv    # Top 6 selected features
├── notebooks/
│   ├── 01_data_preprocessing.ipynb    # Data cleaning and preparation
│   ├── 02_pca_analysis.ipynb          # Principal Component Analysis
│   ├── 03_feature_selection.ipynb     # Feature importance and selection
│   ├── 04_supevised_learning.ipynb    # Model training and comparison
│   ├── 05_unsupervised_learning.ipynb # Clustering analysis
│   ├── 06_hyperparameter_turning.ipynb # Model optimization
│   └── 07_model_export.ipynb          # Final model export
├── models/
│   └── final_model.pkl                # Trained Random Forest pipeline
├── results/
│   └── evaluation_metrics.txt         # Comprehensive model metrics
└── ui/
    └── app.py                         # Streamlit web application
```

## 🔬 Machine Learning Pipeline

### 1. Data Preprocessing
- Handle missing values
- Remove duplicates
- Normalize features (StandardScaler)
- Create binary and multiclass targets

### 2. Feature Engineering
- **Feature Selection Methods**:
  - Random Forest feature importance
  - Recursive Feature Elimination (RFE)
  - Chi-Square test
- **Final Selected Features** (6):
  - `thal`: Thalassemia test result
  - `exang`: Exercise induced angina
  - `slope`: ST segment slope
  - `ca`: Major vessels count
  - `cp`: Chest pain type
  - `oldpeak`: ST depression

### 3. Model Development
**Algorithms Tested**:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

**Best Model**: Random Forest Classifier
- **Accuracy**: ~85-88%
- **Hyperparameters**:
  - `n_estimators`: 100
  - `max_depth`: 5
  - `min_samples_split`: 10
  - `min_samples_leaf`: 1
  - `max_features`: 'log2'

### 4. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix
- Cross-validation
- Feature importance analysis

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/nhz1211/ML-Project--Heart-Disease-Prediction.git
cd ML-Project--Heart-Disease-Prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Project : Streamlit Web Application

1. **Navigate to UI folder**
```bash
cd ui
```

2. **Run the Streamlit app**
```bash
streamlit run app.py
```

3. **Access the application** at `http://localhost:8501`

## 🌐 Web Application Features

The interactive Streamlit app provides:

1. **🏠 Prediction Page**
   - Input patient health metrics
   - Real-time risk assessment
   - Probability scores
   - Clinical recommendations

2. **📊 Data Exploration Page**
   - Dataset statistics
   - Disease distribution visualizations
   - Correlation heatmap
   - Feature distribution analysis
   - Feature importance rankings

3. **ℹ️ About Page**
   - Project information
   - Model details
   - Medical disclaimer

## 📈 Results

Detailed evaluation metrics are available in `results/evaluation_metrics.txt`

**Summary**:
- **Best Model**: Random Forest (Tuned)
- **Test Accuracy**: 85-88%
- **F1-Score**: 85-88%
- **AUC-ROC**: 90-93%

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 82-85% | 83-86% | 81-84% | 82-85% |
| Decision Tree | 75-78% | 76-79% | 74-77% | 75-78% |
| Random Forest (Tuned) | **85-88%** | **86-89%** | **84-87%** | **85-88%** |
| SVM (Tuned) | 84-87% | 85-88% | 83-86% | 84-87% |

## 🔍 Clustering Analysis

**Unsupervised Learning Results**:
- K-Means (K=2): Silhouette Score ~0.20-0.30
- Hierarchical Clustering: Similar performance
- Both methods show reasonable alignment with actual disease labels

## 🛠️ Technologies Used

- **Python 3.13**
- **Data Science**: Pandas, NumPy, SciPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Streamlit
- **Model Persistence**: Joblib
- **Development**: Jupyter Notebook

## 👥 Contributors

**Ain Shams University - Faculty of Engineering**
- **Malak Ossama** (22P0052)
- **Nour Hossam** (2201386)

**Course**: [CSE381] Introduction to Machine Learning  
**Instructors**: Dr. Alaa Mahmoud Hamdy, Eng. George Welson

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. It should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## 📝 License

This project is created for educational purposes as part of university coursework.
