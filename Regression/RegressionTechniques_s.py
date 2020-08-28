import pickle
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn import linear_model
from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def linear_multiple_regression(x_tr, x_ts, y_tr, y_ts, filename='linear_regression'):
    print("Linear Multiple Regression")
    x_train = x_tr
    x_test = x_ts
    y_train = y_tr
    y_test = y_ts

    cls = linear_model.RidgeCV(alphas=np.logspace(-9, 9, 19), normalize=True)
    #cls = linear_model.LinearRegression(normalize=True)

    str_accuracy = "Multiple Regression Accuracy: "
    str_mean_s_error = "Multiple Regression Mean Square Error: "

    if x_tr.ndim == 1:
        x_train = np.expand_dims(x_train, axis=1)
        x_test = np.expand_dims(x_test, axis=1)
        y_train = np.expand_dims(y_train, axis=1)
        y_test = np.expand_dims(y_test, axis=1)
        str_accuracy = "Linear Regression Accuracy: "
        str_mean_s_error = "Linear Regression Mean Square Error: "

    print("Fitting...")
    cls.fit(x_train, y_train)

    # Save The Model into File
    filename_model = f'Trained Models\{filename}.sav'
    pickle._dump(cls, open(filename_model, 'wb'))

    #print("Best Alpha : ", cls.alpha_)

    print("Predicting...")
    prediction = cls.predict(x_test)

    print(str_accuracy, round(cls.score(x_test, y_test)*100), "%")
    print(str_mean_s_error, metrics.mean_squared_error(np.asarray(y_test), prediction), "\n\n")

    '''ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
    sns.distplot(prediction, hist=False, color="b", label="Predicted Values", ax=ax1)
    plt.show()'''

    '''fig, ax = plt.subplots()
    ax.scatter(y_test, prediction)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('measured')
    ax.set_ylabel('predicted')
    plt.show()'''


def poly_regression(x_tr, x_ts, y_tr, y_ts, degree, filename='Poly_regression'):
    print("Polynomial Regression")
    x_train = x_tr
    x_test = x_ts
    y_train = y_tr
    y_test = y_ts

    poly_features = PolynomialFeatures(degree=degree)
    x_train_poly = poly_features.fit_transform(x_train)
    poly_model = linear_model.RidgeCV(alphas=np.logspace(-9, 9, 19), normalize=True)
    #poly_model = linear_model.LinearRegression(normalize=True)

    print("Fitting...")
    poly_model.fit(x_train_poly, y_train)

    # Save The Model into File
    filename_model = f'Trained Models\{filename}.sav'
    pickle._dump(poly_model, open(filename_model, 'wb'))

    #print("Best Alpha : ", poly_model.alpha_)

    print("Predicting...")
    prediction = poly_model.predict(poly_features.fit_transform(x_test))

    print("Poly Accuracy: ", round(poly_model.score(poly_features.fit_transform(x_test), y_test)*100), "%")
    print('Poly Mean Square Error', metrics.mean_squared_error(y_test, prediction), "\n\n")
    print("First Value of Test Samples' Actual Output: ", np.asarray(y_test)[0])
    print("First Value of Test Samples' Predicted Output: ", prediction[0])

    '''fig, ax = plt.subplots()
    ax.scatter(y_test, prediction)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('measured')
    ax.set_ylabel('predicted')
    plt.show()'''


def ready_linear_multiple_regression(x_ts, y_ts, filename='linear_regression'):
    print("(Ready) Linear Multiple Regression")
    x_test = x_ts
    y_test = y_ts

    # Load The Model from File
    filename_model = f'Trained Models\{filename}.sav'
    model = pickle.load(open(filename_model, 'rb'))

    str_accuracy = "Multiple Regression Accuracy: "
    str_mean_s_error = "Multiple Regression Mean Square Error: "

    if x_ts.ndim == 1:
        x_test = np.expand_dims(x_test, axis=1)
        y_test = np.expand_dims(y_test, axis=1)
        str_accuracy = "Linear Regression Accuracy: "
        str_mean_s_error = "Linear Regression Mean Square Error: "


    print("Predicting...")
    prediction = model.predict(x_test)

    print(str_accuracy, round(model.score(x_test, y_test)*100), "%")
    print(str_mean_s_error, metrics.mean_squared_error(np.asarray(y_test), prediction), "\n\n")

    ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
    sns.distplot(prediction, hist=False, color="b", label="Predicted Values", ax=ax1)
    plt.show()

    '''fig, ax = plt.subplots()
    ax.scatter(y_test, prediction)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('measured')
    ax.set_ylabel('predicted')
    plt.show()'''


def ready_poly_regression(x_ts, y_ts, degree, filename='Poly_regression'):
    print("(Ready) Polynomial Regression")
    x_test = x_ts
    y_test = y_ts

    # Load The Model from File
    filename_model = f'Trained Models\{filename}.sav'
    poly_model = pickle.load(open(filename_model, 'rb'))

    poly_features = PolynomialFeatures(degree=degree)


    print("Predicting...")
    prediction = poly_model.predict(poly_features.fit_transform(x_test))

    print("Poly Accuracy: ", round(poly_model.score(poly_features.fit_transform(x_test), y_test)*100), "%")
    print('Poly Mean Square Error', metrics.mean_squared_error(y_test, prediction), "\n\n")
    print("First Value of Test Samples' Actual Output: ", np.asarray(y_test)[0])
    print("First Value of Test Samples' Predicted Output: ", prediction[0])

    ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
    sns.distplot(prediction, hist=False, color="b", label="Predicted Values", ax=ax1)
    plt.show()

    '''fig, ax = plt.subplots()
    ax.scatter(y_test, prediction)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    ax.set_xlabel('measured')
    ax.set_ylabel('predicted')
    plt.show()'''
