import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Function 1: Load, clean, and prepare dataset
def load_and_prepare_data(path="employee_attrition.csv"):
    df = pd.read_csv(path)

    print("\n📊 Avg Monthly Hours - Mean: {:.2f}, Max: {:.2f}".format(
        df['average_monthly_hours'].mean(), df['average_monthly_hours'].max()))

    for col in ['department', 'salary_level']:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col])

    scaler = StandardScaler()
    df[df.columns.difference(['left'])] = scaler.fit_transform(df[df.columns.difference(['left'])])

    print("✅ Dataset loaded and preprocessed.")
    return df

# Function 2: Hypothesis function demo (returns probability)
def hypothesis_demo():
    x_sample = np.array([0.5, -1.2, 0.8])
    weights = np.array([1.5, -0.8, 2.0])
    bias = 0.3

    z = np.dot(weights, x_sample) + bias
    h_x = 1 / (1 + np.exp(-z))

    print(f"\n📐 Hypothesis h(x) = sigmoid(w·x + b)")
    print(f"🧮 z = {z:.4f}")
    print(f"🔢 Probability that employee will leave = {h_x:.4f}")

# Function 3: Sigmoid activation demo
def sigmoid_demo():
    z = 2.0
    sigmoid = 1 / (1 + np.exp(-z))
    print(f"\n🧠 Sigmoid(2.0) = {sigmoid:.4f}")

# Function 4: Custom log loss cost function
def cost_function(y_true, y_pred_prob):
    epsilon = 1e-15
    y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))

# Function 5: Train and evaluate the model
def train_and_evaluate(X_train, y_train, X_test, y_test, path="attrition_model.pkl"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    joblib.dump(model, path)
    print(f"\n✅ Model trained and saved to '{path}'")

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    cost = cost_function(y_test.values, y_pred_prob)

    print(f"\n🎯 Log Loss (Custom Cost): {cost:.4f}")
    print("📌 Sample Predictions:", y_pred[:10])

# --------- Main Logic ---------
if __name__ == "__main__":
    df = load_and_prepare_data("employee_attrition.csv")

    hypothesis_demo()
    sigmoid_demo()

    X = df.drop(columns=['left'])
    y = df['left']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_and_evaluate(X_train, y_train, X_test, y_test)
