# Linear Regression Model
# Importing the libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# Question 1
# Importing the data (20190449: inputdata9)
Mydata = np.loadtxt('C:\\Users\\RajaI\\Desktop\\amna\\inputdata9.csv',
                    delimiter=",", dtype=str, usecols=(0, 1))
# print(Mydata)

# Data Cleaning
# Remove first columns of inout data with headers
Mydata = np.delete(Mydata, (0), axis=0)

# Data Splitting into X and Y array
X = Mydata[:, 0]
# print(X)
Y = Mydata[:, 1]
# print(Y)

# -----------------------------------------------------------------------------

# Question 2
#  Simple Scatter Plot using X and Y
plt.scatter(X, Y, s=60, c='red', edgecolor='red', linewidth=1)
plt.title('Rainfall vs Productivity', fontsize=12)
plt.xlabel('Rainfall', fontsize=12)
plt.ylabel('Productivity', fontsize=12)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------

# Question 3
# Reshaping is great if you passed a NumPy array.
# Reshape your data either using array.reshape(-1, 1)
X = Mydata[:, 0].astype(float).reshape(-1, 1)
Y = Mydata[:, 1].astype(float)

# Model Buiding and fit X and Y
model = LinearRegression()
model = model.fit(X, Y)

# Model Testing and check whether the model works satisfactorily
# Obtain the properties of the model
r_sq = model.score(X, Y)
print(f"coefficient of determination: {r_sq}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")

# -----------------------------------------------------------------------------

# Question 4
# Model Predictions with existing data
y_pred = model.predict(X)
# print(f"predicted response:\n{y_pred}")

# Scatter Plot to show Linear Regression on either existing
plt.scatter(X, Y, s=60, c='red', edgecolor='red', linewidth=1)
plt.plot(X, y_pred, color='blue')
plt.title('Scatter Plot with Linear Regression Model', fontsize=12)
plt.xlabel('Rainfall', fontsize=12)
plt.ylabel('Productivity', fontsize=12)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------

# Question 5
# Calculate Prodcutivity Coefiiceint Prediction
X_mm = 275.0
x_pred = np.array([X_mm]).reshape(-1, 1)  # reshaping data
# Coefiiceint Prediction of X_mm = 275.0
x_pred = model.predict(x_pred)
print(f"coefficient for X_mm = 275.0 : {x_pred}")

# Another way for calculating Coefiiceint Prediction of X_mm = 275.0
# productivity_coef = model.coef_ * X_mm + model.intercept_
# print('Productivity coefficient is=', productivity_coef)

# Plotting Data and Linear Regression with X_mm
plt.scatter(X, Y, s=60, c='red', edgecolor='red', linewidth=1)
plt.plot(X, y_pred, color='blue')
# plotting the predicting value
plt.plot(X_mm, x_pred, c='cyan', marker='^', markersize=20)
plt.text(X_mm, x_pred, x_pred)
plt.title('Scatter Plot with Prodcutivity Coefiiceint Prediction', fontsize=12)
plt.xlabel('Rainfall', fontsize=12)
plt.ylabel('Productivity', fontsize=12)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.show()
