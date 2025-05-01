import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, XGBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Generate synthetic dataset for hydraulic failure classification
np.random.seed(42)
num_samples = 2200
df = pd.DataFrame({
    'PS1_mean': np.random.normal(100, 5, num_samples),
    'EPS1_max': np.random.normal(5000, 300, num_samples),
    'FS1_std': np.random.normal(2, 0.5, num_samples),
    'TS1_mean': np.random.normal(40, 3, num_samples),
    'CP_max': np.random.normal(12, 1.2, num_samples),
    'SE_mean': np.random.normal(85, 5, num_samples),
    'CoolerCondition': np.random.choice([3, 20, 100], num_samples)
})

X = df.drop(columns=['CoolerCondition'])
y = df['CoolerCondition']

# Define classifiers
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBoostClassifier(random_state=42),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline and evaluation
results = {}
for name, model in models.items():
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }

# Convert results to DataFrame for display
results_df = pd.DataFrame(results).T.sort_values(by='F1 Score', ascending=False)
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)
results_df.head()