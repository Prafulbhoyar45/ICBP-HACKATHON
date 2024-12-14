import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load datasets
cnc_datasets = {
    "CNC01": pd.read_excel('CNC001.xlsx'),
    "CNC02": pd.read_excel('CNC002.xlsx'),
    "CNC03": pd.read_excel('CNC003.xlsx'),
    "CNC04": pd.read_excel('CNC004.xlsx'),
    "CNC05": pd.read_excel('CNC005.xlsx'),
    "CNC06": pd.read_excel('CNC006.xlsx'),
    "CNC07": pd.read_excel('CNC007.xlsx')
}

# Train models for each CNC machine
def train_model(data):
    # Identify target and features
    X = data.drop(columns=['Maintenance Required'])
    y = data['Maintenance Required']
    
    # Encode categorical columns in X
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Encode target variable if needed
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Decision Tree Classifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

# Train and save models for all CNC datasets
for cnc, dataset in cnc_datasets.items():
    model = train_model(dataset)
    # Save each model as a separate file
    with open(f"{cnc}_model.pkl", "wb") as f:
        pickle.dump(model, f)

print("All models have been trained and saved.")
