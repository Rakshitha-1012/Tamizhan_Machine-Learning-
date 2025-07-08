
# Loan Eligibility Predictor - Full Code

# 📦 Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# 📁 Load or Create Sample Dataset
data = pd.DataFrame({
    'age': [25, 40, 35, np.nan, 45, 52, 23, 30],
    'income': [50000, 80000, 60000, 75000, np.nan, 120000, 45000, 54000],
    'education': ['graduate', 'not_graduate', 'graduate', 'graduate', 'not_graduate', 'graduate', 'not_graduate', 'graduate'],
    'credit_score': [700, 650, 600, 710, 680, np.nan, 590, 620],
    'loan_approved': ['yes', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'no']
})

# 🧹 Data Preprocessing
# Handle missing values
imputer = SimpleImputer(strategy='mean')
data[['age', 'income', 'credit_score']] = imputer.fit_transform(data[['age', 'income', 'credit_score']])

# Encode categorical features
le_edu = LabelEncoder()
data['education'] = le_edu.fit_transform(data['education'])  # graduate=1, not_graduate=0

le_target = LabelEncoder()
data['loan_approved'] = le_target.fit_transform(data['loan_approved'])  # yes=1, no=0

# 🧾 Features and Target
X = data.drop('loan_approved', axis=1)
y = data['loan_approved']

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔄 Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🤖 Model 1: Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# 🤖 Model 2: Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 📊 Evaluation Function
def evaluate_model(name, y_test, y_pred, model, X_test):
    print(f"\n📌 Evaluation Report for {name}:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # ROC Curve
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label=f'{name} ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title(f'{name} - ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# 📈 Evaluate Models
evaluate_model("Logistic Regression", y_test, y_pred_log, log_model, X_test)
evaluate_model("Random Forest", y_test, y_pred_rf, rf_model, X_test)

# 🧪 Predict for New Applicant
new_applicant = pd.DataFrame({
    'age': [29],
    'income': [60000],
    'education': le_edu.transform(['graduate']),
    'credit_score': [690]
})
new_applicant_scaled = scaler.transform(new_applicant)
log_result = log_model.predict(new_applicant_scaled)
rf_result = rf_model.predict(new_applicant_scaled)

print("✅ Logistic Regression Prediction:", le_target.inverse_transform(log_result)[0])
print("✅ Random Forest Prediction:", le_target.inverse_transform(rf_result)[0])
 
