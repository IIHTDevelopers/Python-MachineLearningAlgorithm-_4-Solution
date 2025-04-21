import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

# Function 1: Load and preprocess the dataset
def load_and_preprocess(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    df = df.dropna()
    print("📚 Student data loaded and cleaned.")
    return df

# Function 2: Show basic stats
def show_key_stats(df):
    hours_std = df['hours_studied'].std()
    max_previous_score = df['previous_score'].max()
    print(f"\n📊 Standard Deviation of Study Hours: {hours_std:.2f}")
    print(f"🏅 Max Previous Score: {max_previous_score}")

# Function 3: Prepare data
def prepare_data(df, features, target):
    X = df[features]
    y = df[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print("\n🧪 Data prepared and split.")
    return X_train, X_test, y_train, y_test, scaler

# Function 4: Train and save model
def train_and_save_model(X_train, y_train, model_path="student_score_model.pkl"):
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"\n✅ Model trained and saved to '{model_path}'")
    return model

# Function 5: Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\n🎯 Mean Squared Error: {mse:.2f}")
    print("📈 Sample Predictions:", y_pred[:5])

# ---- MAIN SCRIPT ----
if __name__ == "__main__":
    features = ['hours_studied', 'previous_score', 'assignments_completed']
    target = 'final_score'

    df = load_and_preprocess("student_scores.csv")
    show_key_stats(df)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, features, target)
    model = train_and_save_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
