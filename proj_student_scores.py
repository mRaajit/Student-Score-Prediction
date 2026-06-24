import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Load dataset
df = pd.read_csv("student_scores.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Dataset information
print("\nShape:", df.shape)

# Scatter plot
plt.scatter(df["Hours"], df["Scores"])
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Hours vs Scores")
plt.show()

# Features and target
X = df[["Hours"]]
y = df["Scores"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions on test data
predictions = model.predict(X_test)

# Evaluate model
r2 = r2_score(y_test, predictions)

print("\nR² Score:", r2)

# Custom prediction
hours = 7

predicted_score = model.predict([[hours]])

print(f"\nPredicted score for {hours} study hours: {predicted_score[0]:.2f}")

# Equation of line
print("\nModel Equation:")
print(
    f"Score = {model.coef_[0]:.2f} * Hours + {model.intercept_:.2f}"
)