import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("emails.csv")

# Data info and head
df.info()
df.head()

# Define feature matrix X and target vector y
X = df.iloc[:, 1:-1]
y = df['Prediction']

# Preprocess the dataset
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Function to build and evaluate models
def make_model(X, y, model):
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
    
    # Build and train model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = pipeline.predict(X_test)
    cr = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return pipeline, cr, cm, acc, precision, recall, f1

# Models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42)
}

# Initialize dictionaries to store results
model_results = {
    'accuracy': {},
    'precision': {},
    'recall': {},
    'f1_score': {}
}

# Evaluate each model
for model_name, model in models.items():
    _, cr, cm, acc, precision, recall, f1 = make_model(X, y, model)
    print(f"Results for {model_name}:")
    print(cr)
    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    ConfusionMatrixDisplay(cm).plot()
    plt.title(model_name)
    plt.show()
    
    # Store results
    model_results['accuracy'][model_name] = acc
    model_results['precision'][model_name] = precision
    model_results['recall'][model_name] = recall
    model_results['f1_score'][model_name] = f1

# Determine the best model for each metric
best_models = {metric: max(results, key=results.get) for metric, results in model_results.items()}

print("Best models:")
for metric, model in best_models.items():
    print(f"Best model for {metric}: {model} with {model_results[metric][model]:.4f}")

# Cross-validation
for model_name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation scores for {model_name}: {scores}")
    print(f"Mean accuracy: {scores.mean()} \n")

# Hyperparameter tuning
param_grid = {
    'Logistic Regression': {'classifier__C': [0.1, 1, 10]},
    'Decision Tree': {'classifier__max_depth': [5, 10, 20]},
    'Support Vector Machine': {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']}
}

for model_name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_} \n")

# PCA for dimensionality reduction
from sklearn.decomposition import PCA

pca = PCA(n_components=1000)
X_new = pca.fit_transform(X)

# Evaluate each model after PCA
for model_name, model in models.items():
    _, cr, cm, acc, precision, recall, f1 = make_model(X_new, y, model)
    print(f"Results for {model_name} after PCA:")
    print(cr)
    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"{model_name} after PCA")
    plt.show()
    
    # Store results
    model_results['accuracy'][model_name] = acc
    model_results['precision'][model_name] = precision
    model_results['recall'][model_name] = recall
    model_results['f1_score'][model_name] = f1

# Determine the best model for each metric after PCA
best_models_pca = {metric: max(results, key=results.get) for metric, results in model_results.items()}

print("Best models after PCA:")
for metric, model in best_models_pca.items():
    print(f"Best model for {metric} after PCA: {model} with {model_results[metric][model]:.4f}")