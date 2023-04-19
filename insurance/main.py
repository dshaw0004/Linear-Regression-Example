import pandas as pd  # pandas for extracting data from csv file
from sklearn.linear_model import LinearRegression  # Linear regression model by scikit-learn
from sklearn.model_selection import train_test_split  # for splitting data for training and testing
import numpy as np

df = pd.read_csv("insurance.csv")  # creating a pandas dataframe of the data of csv file

# creating a column in the dataframe of name 'male'.
# it will have value of 1 if corresponding value of 'sex' column is 'male' and 0 for 'female'
df['male'] = df['sex'].map({'male': 1, 'female': 0})

# converting value of 'smoker' column into binary
# 1 for smoker and 0 for non-smoker
df['smoker'].replace(['yes', 'no'], [1, 0], inplace=True)

# changing value of 'region' column into number
# northeast = 0, northwest = 1, southeast = 2, southwest = 3
df['region'].replace(['northeast', 'northwest', 'southeast', 'southwest'], [0, 1, 2, 3], inplace=True)

# gathering all independent variables value in a single variable
x = df[['male', 'smoker', 'region', 'age', 'bmi', 'children']].values
# dependent variable
y = df['charges'].values

# splitting data for training and testing
# test_size defines the size of test data. 0.2 means 20%
trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2)

# creating a instance of LinearRegression model
model = LinearRegression()

# fitting the training data to the model
model.fit(trainX, trainY)

# checking the accuracy of the model using test data
print(model.score(testX, testY))

# predicting the 'charges' for custom value

# for testing one set of custom value at a time
custom_single_value = np.array([1, 1, 0, 27, 32.0, 0]).reshape(1, -1)
print(model.predict(custom_single_value))

# for testing more than one custom values at a time
custom_value = pd.DataFrame([[1, 0, 2, 35, 36.9, 1], [0, 1, 1, 33, 27.14, 2]])
print(model.predict(custom_value.values))

# if you have more a lot of data for testing use this method
# write down the data in the 'testing data.csv' file
testing_data = pd.read_csv("testing data.csv")
print(model.predict(testing_data.values))
