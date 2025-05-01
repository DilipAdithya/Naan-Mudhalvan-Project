import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Generate Synthetic Dataset
np.random.seed(42)
num_samples = 500
data = pd.DataFrame({
    'Material_Thickness_mm': np.random.uniform(1.0, 50.0, num_samples),
    'Press_Type': np.random.choice(['Mechanical', 'Hydraulic', 'Pneumatic'], num_samples),
    'Pressure_applied_Bar': np.random.uniform(100, 500, num_samples),
    'Operation_time_s': np.random.uniform(2, 60, num_samples),
    'Hydraulic_fluid_level_%': np.random.uniform(20, 100, num_samples),
    'Cylinder_size_cm2': np.random.uniform(10, 200, num_samples),
    'Number_of_strokes': np.random.randint(1, 10, num_samples),
    'Tooling_Type': np.random.choice(['Standard', 'Custom', 'Advanced'], num_samples),
    'Temperature_C': np.random.uniform(15, 80, num_samples),
    'Maintenance_Status': np.random.choice(['Up-to-date', 'Due', 'Overdue'], num_samples),
})

# Step 2: Create Target Variable
data['Power_Consumption_kWh'] = (
    0.1 * data['Material_Thickness_mm'] +
    0.05 * data['Pressure_applied_Bar'] +
    0.3 * data['Operation_time_s'] +
    0.02 * data['Cylinder_size_cm2'] +
    0.5 * data['Number_of_strokes'] +
    0.01 * data['Temperature_C'] +
    np.random.normal(0, 5, num_samples)
)

# Step 3: Preprocessing and Model Training
X = data.drop(columns=['Power_Consumption_kWh'])
y = data['Power_Consumption_kWh']
categorical_cols = ['Press_Type', 'Tooling_Type', 'Maintenance_Status']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first'), categorical_cols)
])

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    results[name] = {
        'R2 Score': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }

# Display Results
results_df = pd.DataFrame(results).T.sort_values(by='R2 Score', ascending=False)
print(results_df)