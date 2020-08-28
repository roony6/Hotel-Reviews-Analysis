import os
import pandas as pd
from PreProcessing_s import *
from RegressionTechniques_s import *
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns


data = pd.read_csv('Hotel_Reviews.csv')

data_train, data_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)


data_train = pd.DataFrame(data_train)
data_test = pd.DataFrame(data_test)

if os.path.isfile('DataToTrain.pkl'):
    print("Data Train is found")
    all_data_train = pd.read_pickle('DataToTrain.pkl')
    Y_train = all_data_train['Reviewer_Score']
    X_train = all_data_train.drop(columns=['Reviewer_Score'])
else:
    print("Data Train is not found")
    X_train, Y_train = ready_data(data_train)
    all_data_train = pd.concat([X_train, Y_train], axis=1)
    all_data_train.to_pickle('DataToTrain.pkl')

if os.path.isfile('DataToTest.pkl'):
    print("Data Test is found")
    all_data_test = pd.read_pickle('DataToTest.pkl')
    Y_test = all_data_test['Reviewer_Score']
    X_train = all_data_train.drop(columns=['Reviewer_Score'])
else:
    print("Data Test is not found")
    X_test, Y_test = ready_data(data_test, test=True)
    all_data_test = pd.concat([X_test, Y_test], axis=1)
    all_data_test.to_pickle('DataToTest.pkl')


'''data1_train = all_data_train.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 21]]
data2_train = all_data_train.iloc[:, [8, 9, 10, 11, 12, 13, 19, 20, 21]]

data1_test = all_data_test.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 21]]
data2_test = all_data_test.iloc[:, [8, 9, 10, 11, 12, 13, 19, 20, 21]]'''

# data1 corr plot
'''plt.subplots(figsize=(5, 5))
corr = data1_train.corr()
sns.heatmap(corr, annot=True)
plt.show()'''

# data2 corr plot
'''plt.subplots(figsize=(5, 5))
corr = data2_train.corr()
sns.heatmap(corr, annot=True)
plt.show()'''


# top_features corr
corr = all_data_train.corr()
top_features = corr.index[abs(corr['Reviewer_Score']) > 0.2]
print('Top Features', top_features)

# top_features corr plot
plt.subplots(figsize=(5, 5))
top_corr = all_data_train[top_features].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

# split the data
new_X_train = all_data_train[top_features]
new_X_train = new_X_train.drop(columns=['Reviewer_Score'])
# Review_Total_Negative_Word_Counts with  NLTK
# Negative_Review with no NLTK
linear_X_train = all_data_train['Negative_Review']

new_X_test = all_data_test[top_features]
new_X_test = new_X_test.drop(columns=['Reviewer_Score'])
linear_X_test = all_data_test['Negative_Review']


# Training Phase
# Linear Regression
linear_multiple_regression(linear_X_train, linear_X_test, Y_train, Y_test)

# Multiple Regression
linear_multiple_regression(new_X_train, new_X_test, Y_train, Y_test, 'multiple_regression')

# Polynomial Regression
poly_regression(new_X_train, new_X_test, Y_train, Y_test, 4)

# ------------------------------------------------------------

# Testing Phase
'''test = pd.read_excel('hotel_reviews_regression_test.xlsx')
X_test, Y_test = ready_data(test)
all_data_test = pd.concat([X_test, Y_test], axis=1)

new_X_test = all_data_test[top_features]
new_X_test = new_X_test.drop(columns=['Reviewer_Score'])
linear_X_test = all_data_test['Negative_Review']


ready_linear_multiple_regression(linear_X_test, Y_test)

ready_linear_multiple_regression(new_X_test, Y_test, 'multiple_regression')

ready_poly_regression(new_X_test, Y_test, degree=4)'''