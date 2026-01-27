import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("ML Assignment 2 – Classification Models")
st.write("Upload test data and select a trained model to view evaluation metrics.")

# Model selection
model_name = st.selectbox(
    "Select Classification Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

# Upload test dataset
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    # MNIST CSV has NO header
    data = pd.read_csv(uploaded_file, header=None)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(data.head())

    st.info("MNIST dataset detected: First column is treated as target label")

    # First column = label, rest = features
    y_true = data.iloc[:, 0]
    X = data.iloc[:, 1:]

    # Model paths
    model_path_map = {
        "Logistic Regression": "model/logistic.pkl",
        "Decision Tree": "model/decision_tree.pkl",
        "KNN": "model/knn.pkl",
        "Naive Bayes": "model/naive_bayes.pkl",
        "Random Forest": "model/random_forest.pkl",
        "XGBoost": "model/xgboost.pkl",
    }

    try:
        # Load model
        model = joblib.load(model_path_map[model_name])

        # Predictions
        y_pred = model.predict(X)

        # Evaluation Metrics
        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
        col2.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.4f}")
        col3.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("F1 Score", f"{f1_score(y_true, y_pred, average='weighted'):.4f}")
        col5.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)

        # Classification Report
        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred))

    except FileNotFoundError:
        st.error("Model file not found. Please upload trained model files (.pkl) in the model folder.")
