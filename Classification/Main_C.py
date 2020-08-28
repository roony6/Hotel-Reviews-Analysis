import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PreProcessing_C import *
from ClassificationTechniques import *
from sklearn.decomposition import PCA
from VisualPlot import *


def train_data(X_train, X_test, Y_train, Y_test):
    # logistic regression
    logistic_regression(X_train, X_test, Y_train, Y_test)
    # decision tree
    decision_tree(X_train, X_test, Y_train, Y_test)
    # knn
    knn(X_train, X_test, Y_train, Y_test)
    # random forest
    random_forest(X_train, X_test, Y_train, Y_test)
    # One vs One SVM
    svm_one_vs_one_class(X_train, X_test, Y_train, Y_test)
    # One vs Rest SVM
    svm_one_vs_rest_class(X_train, X_test, Y_train, Y_test)


def classification_train(X_train, X_test, Y_train, Y_test, pca=0):
    if pca:
        pca = PCA(n_components=0.99)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    train_data(X_train, X_test, Y_train, Y_test)


def test_trained_data(X_test, Y_test, X_train, pca=0):
    if pca:
        pca = PCA(n_components=0.99)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    ready_logistic_regression(X_test, Y_test)
    ready_decision_tree(X_test, Y_test)
    ready_knn(X_test, Y_test)
    ready_random_forest(X_test, Y_test)
    ready_svm_one_vs_one_class(X_test, Y_test)
    ready_svm_one_vs_rest_class(X_test, Y_test)

data = pd.read_csv('Hotel_Reviews_Milestone_2.csv')
# 414736
data = data.iloc[:414736, :]

print(data.columns)

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
    print("READY DATA <Train>")
    X_train, Y_train = ready_data(data_train) # Preprocessing
    all_data_train = pd.concat([X_train, Y_train], axis=1)
    all_data_train.to_pickle('DataToTrain.pkl')

if os.path.isfile('DataToTest.pkl'):
    print("Data Test is found")
    all_data_test = pd.read_pickle('DataToTest.pkl')
    Y_test = all_data_test['Reviewer_Score']
    X_test = all_data_test.drop(columns=['Reviewer_Score'])
else:
    print("Data Test is not found")
    print("\n\nREADY DATA <Test>")
    X_test, Y_test = ready_data(data_test) # Preprocessing
    all_data_test = pd.concat([X_test, Y_test], axis=1)
    all_data_test.to_pickle('DataToTest.pkl')

print(" ----------- ")

#classification_train(X_train, X_test, Y_train, Y_test, 1)
#test_trained_data(X_test, Y_test, X_train, 1)

'''show_bar_graph('Models_Train_Time')
show_bar_graph('Models_Test_Time')
show_bar_graph('Models_Acc_Time')'''

# hotel_reviews_classification_test_shuffled
# New Samples Test
test = pd.read_excel('hotel_reviews_classification_test_shuffled.xlsx')
X_test, Y_test = ready_data(test)
test_trained_data(X_test, Y_test, X_train, 0)