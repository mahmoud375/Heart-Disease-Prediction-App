# 🩺 Heart Disease Prediction - Full Machine Learning Pipeline

[![Project Status: Completed](https://img.shields.io/badge/status-completed-brightgreen.svg)](https://shields.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heart-disease-prediction-app-1.streamlit.app/)

A comprehensive machine learning project that predicts the likelihood of heart disease based on the UCI Heart Disease dataset. This project covers the entire ML lifecycle, from data preprocessing and exploratory data analysis to model training, evaluation, and deployment as an interactive web application.

---

## 🚀 Live Demo

You can access the live interactive application here:

**[➡️ View Live Streamlit App](https://heart-disease-prediction-app-1.streamlit.app/)**

---

## 📋 Project Workflow

This project follows a structured machine learning pipeline:

1.  **Data Preprocessing & Cleaning:** Loaded the dataset, handled missing values with imputation, performed one-hot encoding for categorical features, and standardized numerical features using `StandardScaler`.
2.  **Exploratory Data Analysis (EDA):** Generated visualizations like correlation heatmaps, histograms, and boxplots to understand data distributions and relationships.
3.  **Feature Selection & Dimensionality Reduction:**
    * Applied **Principal Component Analysis (PCA)** to explore dimensionality reduction.
    * Utilized **Random Forest Feature Importance** and **Recursive Feature Elimination (RFE)** to select the most impactful features for modeling.
4.  **Model Training & Evaluation:**
    * Trained and evaluated multiple classification models:
        * Logistic Regression
        * Decision Tree
        * Random Forest
        * Support Vector Machine (SVM)
    * Evaluated models based on Accuracy, Precision, Recall, F1-Score, and AUC-ROC curves.
5.  **Unsupervised Learning:** Applied **K-Means Clustering** and **Hierarchical Clustering** to discover underlying patterns in the data.
6.  **Hyperparameter Tuning:** Optimized the best-performing model (Random Forest) using `GridSearchCV` to enhance its predictive power.
7.  **Deployment:**
    * Saved the final, optimized model pipeline using `joblib`.
    * Developed an interactive web user interface using **Streamlit**.
    * Deployed the application on **Streamlit Community Cloud** for public access.

---

## 🛠️ Technologies & Tools Used

* **Programming Language:** Python 3.9
* **Libraries:**
    * Pandas & NumPy (Data Manipulation)
    * Scikit-learn (Machine Learning)
    * Matplotlib & Seaborn (Data Visualization)
    * Streamlit (Web App Development)
    * Joblib (Model Persistence)
* **Environment:** Jupyter Notebook, VS Code
* **Deployment:** Streamlit Community Cloud, GitHub

---

## ⚙️ Installation & Setup

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mahmoud375/Heart-Disease-Prediction-App.git
    ```

2.  **Navigate to the project directory:**
    ```bash
    cd Heart-Disease-Prediction-App
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run ui/app.py
    ```
    The application will open in your web browser at `http://localhost:8501`.

---

## 📂 Project File Structure

```
Heart_Disease_Project/
│
├── data/
│   ├── cleaned_heart_disease.csv
│   └── heart+disease/
│       └── processed.cleveland.data
│
├── deployment/
│   └── deployment_log.md
│
├── models/
│   └── final_model.pkl
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
│
├── results/
│   └── evaluation_metrics.txt
│
├── ui/
│   └── app.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 📬 Contact

Created by **Mahmoud Elgendy**

* **Portfolio:** [Mahmoud Elgendy](https://my-portfolio-virid-mu.vercel.app/)
* **LinkedIn:** [Mahmoud Elgendy](https://www.linkedin.com/in/mahmoud-elgendy2003/)

Feel free to reach out with any questions or feedback!