import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

gold_data = pd.read_csv("E:\Media\gold_price\gld_price_data.csv")
# print first 5 rows of dataframe
gold_data.head()
# print last 5 rows of dataframe
gold_data.tail()
# number of rows and column
gold_data.shape
# getting some basic information about the data
gold_data.info()
# checking the number of missing values
gold_data.isnull().sum()
# getting the statistical measure of the dataframe
gold_data.describe()

correlation = gold_data.corr()
# constructing a heat map to understand the correlation
sns.heatmap(correlation, cbar = True, square = True, fmt = '.1f', annot = True, annot_kws = {'size' : 8}, cmap = 'Blues')

X = gold_data.drop(['Date', 'GLD'], axis = 1)
Y = gold_data['GLD']

print(X)
print(Y)
# successfully split features and target data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
# test_size corresponds to the ratio in which training and testing data is split
# so here, 20% of the data goes into testing data and 80% goes into training
# random_state is just like a save state slot of random data, so changing the value also gives random data but set is different

regressor = RandomForestRegressor(n_estimators = 100)
#training the model
regressor.fit(X_train, Y_train)

# prediction on test data
test_data_prediction = regressor.predict(X_test)
print(test_data_prediction)

# checking for R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)

# currently Y_test is a pandas array on which we cannot plot, so we need to convert it into a list first
Y_test = list(Y_test)

plt.plot(Y_test, color = 'blue', label = 'Actual Value')
plt.plot(test_data_prediction, color = 'green', label = "Predicted Value")
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of Values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()