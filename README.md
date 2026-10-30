\# Customer Churn Prediction Using Machine Learning



\## 1. Project Overview



Customer Churn Prediction is a Machine Learning classification project that aims to identify customers who are likely to leave a company.



The project uses customer-related information to analyze patterns, understand customer behavior, and predict whether a customer will churn or remain with the company.



The project covers the complete Machine Learning workflow, including Exploratory Data Analysis (EDA), data preprocessing, model training, hyperparameter tuning, model evaluation, and deployment using Streamlit.



\## 2. Dataset Description



The dataset used in this project, customer\_churn\_classification\_50k.csv, was synthetically generated with the assistance of ChatGPT for educational and Machine Learning practice purposes.



\- \*\*Dataset:\*\* Customer Churn Classification

\- \*\*File:\*\* `customer\_churn\_classification\_50k.csv`

\- \*\*Records:\*\* Approximately 50,250

\- \*\*Features:\*\* 24 columns, including the target variable `churn`

\- \*\*Problem Type:\*\* Binary Classification

\- \*\*Target Variable:\*\* `churn`



The target variable contains two classes:

\- `0` – Not Churn

\- `1` – Churn



The dataset contains customer-related information such as age, income, credit score, account balance, tenure, satisfaction score, transactions, and customer activity.



\## 3. Exploratory Data Analysis (EDA)



Exploratory Data Analysis was performed to understand the dataset, identify patterns, and examine the relationships between different features.



The analysis included:



\- Understanding dataset dimensions using `shape`.

\- Examining column names and data types using `info()`.

\- Generating statistical summaries using `describe()`.

\- Checking missing values and understanding data quality.

\- Analyzing the distribution of the target variable.

\- Identifying class imbalance between churn and non-churn customers.

\- Exploring numerical and categorical features.

\- Analyzing relationships and correlations between numerical variables.

\- Using visualizations to understand feature distributions and patterns.



\*\*Libraries used:\*\* Pandas, NumPy, Matplotlib, and Seaborn.



\## 4. Data Preprocessing



Data preprocessing was performed to prepare the dataset for Machine Learning algorithms.



The following steps were implemented:



\### Data Cleaning

\- Removed `customer\_id` because it is an identifier and not a useful predictive feature.

\- Checked invalid and unrealistic values in relevant columns.

\- Handled missing values using appropriate imputation strategies.



\### Feature Separation

\- Separated independent variables (X) and the target variable (y).

\- Used `churn` as the target variable.



\### Train-Test Split

\- Split the dataset into 80% training data and 20% testing data.

\- Used stratified sampling to maintain the original target-class distribution.

\- Set `random\_state=42` for reproducibility.



\### Feature Transformation

\- Used `SimpleImputer` with median strategy for numerical features.

\- Applied `StandardScaler` to numerical features.

\- Used `SimpleImputer` with most-frequent strategy for categorical features.

\- Applied `OneHotEncoder` with `handle\_unknown='ignore'` to categorical features.



\### ColumnTransformer and Pipeline

\- Used `ColumnTransformer` to apply different preprocessing techniques to numerical and categorical columns.

\- Combined preprocessing and model training using Scikit-learn Pipeline.

\- Ensured preprocessing was fitted on training data during model training to avoid data leakage.



\## 5. Machine Learning Models



Five classification algorithms were implemented and compared using the same dataset and preprocessing workflow.



1\. Logistic Regression

2\. Decision Tree Classifier

3\. Random Forest Classifier

4\. Linear Support Vector Machine (Linear SVC)

5\. K-Nearest Neighbors (KNN)



\## 6. Model Evaluation



The following evaluation metrics were used:



\- \*\*Accuracy:\*\* Measures the overall proportion of correct predictions.

\- \*\*Precision:\*\* Measures how many predicted churn customers actually churned.

\- \*\*Recall:\*\* Measures how many actual churn customers were correctly identified.

\- \*\*F1-score:\*\* Balances precision and recall.

\- \*\*ROC-AUC:\*\* Evaluates the model's ability to distinguish between the two classes.



A confusion matrix was also used to understand True Positives, True Negatives, False Positives, and False Negatives.



Since the dataset has an imbalanced target distribution, accuracy alone was not considered sufficient for model selection.



\## 7. Cross-Validation and Hyperparameter Tuning



To improve model performance and identify suitable hyperparameters:



\- Implemented 5-fold Stratified Cross-Validation.

\- Used `GridSearchCV` and `RandomizedSearchCV` for hyperparameter tuning.

\- Used F1-score as the primary optimization metric.

\- Compared the best cross-validation results across the five models.

\- Evaluated the tuned models on the independent test dataset.



\## 8. Final Model Selection



Random Forest Classifier was selected as the final model based on its test F1-score.



\*\*Final Random Forest Test Results:\*\*



| Evaluation Metric | Score |

|---|---:|

| Accuracy | 76.47% |

| Precision | 21.13% |

| Recall | 44.58% |

| F1-score | 28.68% |

| ROC-AUC | 70.38% |



The selected model and its preprocessing pipeline were saved using Joblib for future predictions.



\## 9. Streamlit Web Application



A Streamlit application was developed to make the trained model accessible through a simple user interface.



\*\*Application features:\*\*



\- Upload customer data in CSV format.

\- Preview uploaded customer records.

\- Predict whether customers are likely to churn.

\- Display churn probabilities.

\- Download prediction results as a CSV file.



The application loads the previously saved model and performs predictions without retraining.



\## 10. Technologies and Libraries Used



\- Python

\- Pandas

\- NumPy

\- Matplotlib

\- Seaborn

\- Scikit-learn

\- Streamlit

\- Joblib

\- Jupyter Notebook



\## 11. Project Structure



```text

Customer-Churn-Prediction/

│

├── data/

│   └── customer\_churn\_classification\_50k.csv

│

├── model/

│   └── churn\_pipeline.joblib

│

├── app.py

├── requirements.txt

├── customer\_churn\_prediction.ipynb

├── README.md

└── .gitignore

```



\## 12. How to Run the Project



\### Step 1: Clone the Repository



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

```



\### Step 2: Navigate to the Project Directory



```bash

cd Customer-Churn-Prediction

```



\### Step 3: Install Dependencies



```bash

pip install -r requirements.txt

```



\### Step 4: Run the Streamlit Application



```bash

python -m streamlit run app.py

```



\### Step 5: Make Predictions



\- Open the Streamlit application in your browser.

\- Upload a customer CSV file containing the required input features.

\- Click the Predict Churn button.

\- View the predictions and churn probabilities.

\- Download the prediction results.



\## 13. Future Improvements



\- Improve minority-class churn detection.

\- Experiment with additional classification algorithms.

\- Explore advanced feature engineering techniques.

\- Improve model generalization using additional validation data.

\- Enhance the Streamlit interface and visualizations.



\## 14. Disclaimer



This project is developed for educational and portfolio purposes. The predictions are based on the trained model and should not be treated as guaranteed customer behavior.

