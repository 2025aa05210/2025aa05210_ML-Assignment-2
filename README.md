**a. Problem Statement**

The goal of this project is to build, test, and deploy multiple machine learning classification models using a real dataset. The project shows the complete machine learning process starting from data preparation, model training, performance evaluation, and finally deploying the models using a Streamlit web application.

**b. Dataset Description**

The dataset used in this project is the MNIST Handwritten Digits Dataset, which is a well-known and publicly available dataset for classification problems.

Dataset Information

Source: Public repository (Kaggle / UCI equivalent)

Total samples: More than 60,000

Number of features: 784 pixel values

Output classes: Digits from 0 to 9

Type of problem: Multi-class classification

Each sample represents a handwritten digit image of size 28×28, converted into numerical pixel values. The dataset was divided into training and testing data to evaluate model performance correctly.

**c. Models Used and Evaluation Metrics**

The following classification models were implemented using the same dataset:

Logistic Regression

Decision Tree

K-Nearest Neighbors (KNN)

Naive Bayes

Random Forest (Ensemble Model)

XGBoost (Ensemble Model)

Evaluation Metrics

**To compare the models, the following metrics were used:**

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

| Model               | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression | 0.9549   | 0.9975 | 0.9567    | 0.9549 | 0.9548   | 0.9461 |
| Decision Tree       | 0.8551   | 0.9111 | 0.8558    | 0.8551 | 0.8545   | 0.8262 |
| KNN                 | 0.8802   | 0.9754 | 0.8883    | 0.8802 | 0.8790   | 0.8578 |
| Naive Bayes         | 0.7703   | 0.9579 | 0.7947    | 0.7703 | 0.7688   | 0.7286 |
| Random Forest       | 0.9216   | 0.9950 | 0.9229    | 0.9216 | 0.9214   | 0.9061 |
| XGBoost             | 0.9396   | 0.9972 | 0.9406    | 0.9396 | 0.9394   | 0.9277 |


Logistic Regression gave very good baseline results with high accuracy and AUC.

Decision Tree captured complex patterns but showed signs of overfitting.

KNN performed well but required more memory, making deployment difficult.

Naive Bayes was fast but less accurate due to its strong assumptions.

Random Forest produced stable and reliable results by combining many trees.

XGBoost gave the best overall performance because of boosting and regularization.

**Streamlit Web Application**

A Streamlit web application was created and deployed to show the model results interactively. The application allows users to:

Upload a test dataset in CSV format

Choose a classification model

View evaluation metrics

See the confusion matrix and classification report

**Live App Link**

https://ml-assignment-2-btbetkqpfq9ltmuhtebnz8.streamlit.app/

**Repository Structure**

ML-Assignment-2/
├── app.py
├── requirements.txt
├── README.md
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── model_training.ipynb



**Deployment**

The project code was uploaded to GitHub

The repository was connected to Streamlit Community Cloud

The application was deployed using app.py

The app was tested successfully

**Live App Link**

https://ml-assignment-2-btbetkqpfq9ltmuhtebnz8.streamlit.app/

**Conclusion**

This project successfully demonstrates the complete machine learning workflow, from training multiple models to deploying them in a live web application. All assignment requirements for ML Assignment 2 have been fulfilled.





