# Stock Predictions Model Using KNN

This project explores predicting stock prices using K-Nearest Neighbors (KNN) algorithm. By analyzing historical stock data for Tata Global, we tackle two main problems: classification and regression. The classification model aims to predict whether the stock price will rise or fall, providing buy (+1) or sell (-1) signals. The regression model, on the other hand, predicts the actual closing price of the stock. Using KNN, we experiment with different hyperparameters to optimize model performance and assess the predictions against actual stock price movements.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Libraries Used](#libraries-used)
3. [Data Collection](#data-collection)
4. [Data Exploration](#data-exploration)
5. [Data Visualization](#data-visualization)
6. [Classification Problem](#classification-problem)
   - [Feature Engineering](#feature-engineering)
   - [Target Variable](#target-variable)
   - [Train-Test Split](#train-test-split)
   - [KNN Classifier](#knn-classifier)
   - [Model Evaluation](#model-evaluation)
   - [Predictions](#predictions)
7. [Regression Problem](#regression-problem)
   - [Target Variable](#target-variable-1)
   - [Train-Test Split](#train-test-split-1)
   - [KNN Regressor](#knn-regressor)
   - [Model Evaluation](#model-evaluation-1)
   - [Predictions](#predictions-1)
8. [Conclusion](#conclusion)

## Prerequisites

Make sure you have the following libraries installed:

```bash
pip install pandas numpy matplotlib quandl scikit-learn
```

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quandl
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score
```

## Data Collection

We are using Quandl to fetch historical stock data for Tata Global:

```python
data = quandl.get('NSE/TATAGLOBAL')
```

## Data Exploration

Display the first few rows of the data:

```python
data.head()
```

## Data Visualization

Plotting the closing price:

```python
plt.figure(figsize=(16,8))
plt.plot(data['Close'], label='Closing Price')
plt.legend()
plt.show()
```

## Classification Problem

We aim to predict whether to buy (+1) or sell (-1) the stock:

### Feature Engineering

Creating new features:

```python
data['Open-Close'] = data['Open'] - data['Close']
data['High-Low'] = data['High'] - data['Low']
data = data.dropna()
X = data[['Open-Close', 'High-Low']]
```

### Target Variable

Define the target variable `Y`:

```python
Y = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)
```

### Train-Test Split

Split the data into training and testing sets:

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
```

### KNN Classifier

Using GridSearchCV to find the best parameter for KNN:

```python
params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
knn = KNeighborsClassifier()
model = GridSearchCV(knn, params, cv=5)
model.fit(X_train, Y_train)
```

### Model Evaluation

Evaluate the accuracy of the model:

```python
accuracy_train = accuracy_score(Y_train, model.predict(X_train))
accuracy_test = accuracy_score(Y_test, model.predict(X_test))
print('Train_data Accuracy: %.2f' % accuracy_train)
print('Test_data Accuracy: %.2f' % accuracy_test)
```

### Predictions

Generate and display predictions:

```python
predictions_classification = model.predict(X_test)
actual_predicted_data = pd.DataFrame({'Actual Class': Y_test, 'Predicted Class': predictions_classification})
actual_predicted_data.head()
```

## Regression Problem

Predicting the closing price of the stock:

### Target Variable

Define the target variable `Y`:

```python
Y = data['Close']
```

### Train-Test Split

Split the data into training and testing sets:

```python
X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = train_test_split(X, Y, test_size=0.25)
```

### KNN Regressor

Using GridSearchCV to find the best parameter for KNN Regressor:

```python
knn_reg = KNeighborsRegressor()
model_reg = GridSearchCV(knn_reg, params, cv=5)
model_reg.fit(X_train_reg, Y_train_reg)
predictions = model_reg.predict(X_test_reg)
```

### Model Evaluation

Calculate the root mean square error (RMSE):

```python
rms = np.sqrt(np.mean(np.power((np.array(Y_test_reg) - np.array(predictions)), 2)))
print('Root Mean Square Error:', rms)
```

### Predictions

Generate and display predictions:

```python
valid = pd.DataFrame({'Actual Class': Y_test_reg, 'Predicted Close Value': predictions})
valid.head()
```

## Conclusion

This project demonstrates the use of KNN for both classification and regression problems to predict stock prices. The classification model predicts buy/sell signals, while the regression model predicts the actual closing prices. Adjusting model parameters and further feature engineering could enhance prediction accuracy.
