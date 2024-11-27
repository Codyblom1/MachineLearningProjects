import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#  we are building a "baby classifier" here. It will guess if a baby is crying or not
# based on three factors: hunger level, sleepiness, and diaper status. Let's see how it goes.

# Step 1: Create some fake data to work with (we'll pretend this is real data)
# Features: [hunger_level, sleepiness_level, diaper_status]
# Labels: 0 = "Not Crying", 1 = "Crying" (simple enough, right?)
data = np.array([
    [0.9, 0.8, 1],  # Definitely crying
    [0.2, 0.1, 0],  # Not crying, chill baby
    [0.8, 0.7, 1],  # Probably crying
    [0.1, 0.2, 0],  # Super relaxed baby
    [0.7, 0.9, 1],  # Hungry and tired, definitely crying
    [0.3, 0.4, 0],  # Happy baby
    [0.6, 0.8, 1],  # Not a happy camper
    [0.2, 0.1, 0],  # Another chill one
])

# Labels (this is what we're trying to predict)
labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])

# Step 2: Split the data into training and testing sets
# We'll use 75% of the data to train the model and 25% to test how good it is.
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

# Step 3: Train a simple logistic regression model
# Honestly, logistic regression is super basic, but it's good enough for a starter project.
model = LogisticRegression()
model.fit(X_train, y_train)  # Train it using the training data

# Step 4: Make predictions
# Time to see how the model does on the test data.
y_pred = model.predict(X_test)

# Step 5: Evaluate the modell
# Lets check the accuracy and print a report to see how well the model is doing.
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Test the model with new data
# Let's throw some new (made-up) data at it to see if it works.
new_data = np.array([
    [0.8, 0.9, 1],  # Probably crying
    [0.1, 0.2, 0],  # Definitely not crying
])
new_predictions = model.predict(new_data)

print("Predictions for new data:", new_predictions)  # 1 means crying, 0 means not crying

"""
Observation: This was fun and pretty straightforward to set up. The accuracy isn't too bad for such a small dataset,
but it would be cool to try this with more realistic data or even add more features. Maybe the next step
could be exploring a different algorithm or visualizing the decision boundaries. Definitely feeling like I learned something new here!
"""