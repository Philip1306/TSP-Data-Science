# Prediction using Supervised ML
# Task 01 by JERIN PHILIP

# Predict the percentage of an student based on the no. of study hours.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

print(s_data.head(10))

# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage', color="red")
plt.xlabel('Hours Studied', color="red")
plt.ylabel('Percentage Score', color="red")
plt.show()

X = s_data.iloc[:, :-1].values
Y = s_data.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print("-----------------------")
print("Training complete.")
print("-----------------------")

# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, Y, color="red")
plt.plot(X, line, color="blue")
plt.show()
# Testing data - In Hours
print("TESTING DATA IN HOURS:")
print(X_test)

# Predicting the scores
y_pred = regressor.predict(X_test)

# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

print("-----------------------")

print("PREDICTED DATA OF THE INPUT TEST DATA:")
print(df)

print("-----------------------")

# What will be predicted score if a student studies for 9.25 hrs/ day?
print("What will be predicted score if a student studies for 9.25 hrs/ day?")
hours = 9.25
pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(pred[0]))

# Evaluating the model
print("Evaluating the model")

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
