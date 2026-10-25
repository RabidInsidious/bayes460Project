######################################################

#  Student Name: Marc-Alexandre Martin-Lauzer

#  Course Code: CSCI 460 -- Fall 2026

#  Assignment Due Date: 10/25/2026

#  GitHub Link: https://github.com/RabidInsidious/bayes460Project

######################################################

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)

# First Task - Prep 

df = pd.read_csv("bank-full.csv", sep=";")

df.to_csv("bank-full-comma.csv", sep=",", index=False)

d = {"yes": 1, "no": 0}

df = df.replace(d)

# Map categorical variables
categorical_columns = [
    'job', 'marital', 'education', 'default',
    'housing', 'loan', 'contact', 'month',
    'poutcome', 'y'
]

for col in categorical_columns:
    # Preserve existing 0/1 mappings
    if col not in ['default', 'housing', 'loan', 'y']:
        df[col] = pd.Categorical(df[col]).codes

# Remove missing values
df = df.dropna()

# Set X and y
X = df.drop(columns=['y'])
y = df['y'].astype(int)

print(df.head())


# Second Task - First Experiment

accuracy_results = []
f1_results = []

for i in range(10):

    # 70% training, 30% testing
    # Stratified random sampling on target variable y
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=i
    )

    # Initialize Gaussian Naive Bayes model
    model = GaussianNB()

    # Train model
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)

    # Measure performance
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    accuracy_results.append(accuracy)
    f1_results.append(f1)

    print(f"\nIteration {i + 1}")
    print("The accuracy of model is:", accuracy)
    print("The F1 Score of model is:", f1)


# Summary of 10 iterations

results = pd.DataFrame({
    "Iteration": range(1, 11),
    "Accuracy": accuracy_results,
    "F1 Score": f1_results
})

print("\nResults for all 10 iterations:")
print(results.to_string(index=False))

print("\nAverage Accuracy:", results["Accuracy"].mean())
print("Standard Deviation Accuracy:", results["Accuracy"].std())

print("\nAverage F1 Score:", results["F1 Score"].mean())
print("Standard Deviation F1 Score:", results["F1 Score"].std())


# Task 3 - Varying Training Data Size

task3_results = []

for train_percent in range(10, 100, 10):

    accuracy_results = []
    f1_results = []

    for i in range(10):

        # Stratified random split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_percent / 100,
            stratify=y,
            random_state=i
        )

        # Initialize and train Gaussian Naive Bayes
        model = GaussianNB()
        model.fit(X_train, y_train)

        # Get predictions
        y_pred = model.predict(X_test)

        # Measure performance
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        accuracy_results.append(accuracy)
        f1_results.append(f1)

        print(
            f"Training: {train_percent}% | "
            f"Iteration: {i + 1} | "
            f"Accuracy: {accuracy:.4f} | "
            f"F1: {f1:.4f}"
        )

    # Average and standard deviation across 10 iterations
    task3_results.append({
        "Training %": train_percent,
        "Testing %": 100 - train_percent,
        "Average Accuracy": pd.Series(accuracy_results).mean(),
        "Std Accuracy": pd.Series(accuracy_results).std(),
        "Average F1": pd.Series(f1_results).mean(),
        "Std F1": pd.Series(f1_results).std()
    })

# Results

results_df = pd.DataFrame(task3_results)

print("\nTask 3: Average Results Across 10 Iterations")
print(results_df.to_string(index=False))

# Performance Plot

plt.figure(figsize=(10, 6))

plt.plot(
    results_df["Training %"],
    results_df["Average Accuracy"],
    marker="o",
    label="Accuracy"
)

plt.plot(
    results_df["Training %"],
    results_df["Average F1"],
    marker="s",
    label="Weighted F1 Score"
)

plt.xlabel("Training Data (%)")
plt.ylabel("Performance Score")
plt.title("Gaussian Naive Bayes Performance vs Training Data Size")

plt.xticks(range(10, 100, 10))
plt.ylim(0, 1)
plt.grid(True)
plt.legend()

plt.show()