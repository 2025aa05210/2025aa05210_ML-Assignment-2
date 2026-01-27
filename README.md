# ML Assignment 2 – Classification Models with Streamlit Deployment

## a. Problem Statement
The objective of this project is to build, evaluate, and deploy multiple machine
learning classification models on a chosen dataset. The project demonstrates
an end-to-end ML workflow including data preprocessing, model training,
performance evaluation, and deployment using a Streamlit web application.

---

## b. Dataset Description
The dataset used for this assignment is a publicly available classification dataset
sourced from a standard open repository (such as Kaggle or UCI).
It satisfies the following conditions:
- Minimum 500 instances
- Minimum 12 features
- Suitable for binary or multi-class classification

The dataset is split into training and testing subsets to evaluate model
performance fairly.

---

## c. Models Used and Evaluation Metrics

The following six classification models were implemented and evaluated on the
same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes Classifier  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)

### Evaluation Metrics Used
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

### Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression |  |  |  |  |  |  |
| Decision Tree |  |  |  |  |  |  |
| KNN |  |  |  |  |  |  |
| Naive Bayes |  |  |  |  |  |  |
| Random Forest |  |  |  |  |  |  |
| XGBoost |  |  |  |  |  |  |

(Fill the values based on your computed results)

---

### Observations on Model Performance

| ML Model | Observation |
|--------|-------------|
| Logistic Regression | Performs well for linearly separable data and provides stable baseline results. |
| Decision Tree | Captures non-linear patterns but may overfit without tuning. |
| KNN | Performance depends heavily on the choice of K and feature scaling. |
| Naive Bayes | Fast and efficient but assumes feature independence. |
| Random Forest | Provides strong performance and robustness by reducing overfitting. |
| XGBoost | Achieves the best performance due to boosting and regularization. |

---

## Streamlit Web Application
An interactive Streamlit web application was developed to:
- Upload test datasets (CSV format)
- Select a classification model
- Display evaluation metrics
- Show confusion matrix or classification report

The app is deployed using **Streamlit Community Cloud**.

---

## Repository Structure

