import time
import pickle
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def logistic_regression(x_tr, x_ts, y_tr, y_ts, filename='logistic_regression'):
    print("\n\nLogistic Regression")
    x_train = x_tr
    x_test = x_ts
    y_train = y_tr
    y_test = y_ts

    model = LogisticRegression(multi_class='auto') # multinomial - ovr(binary)

    print("Fitting Data...")
    trt_strt = time.time()
    model.fit(x_train, y_train)
    trt_end = time.time()
    print("Logistic Regression Train Time", str(round(trt_end - trt_strt, 2)) + " sec")

    with open('Trained Models\Models_Train_Time.txt', 'a') as the_file:
        the_file.write(f'Logistic Regression Train Time: {str(round(trt_end - trt_strt, 2))} Seconds\n')

    # Save The Model into File
    filename_model = f'Trained Models\{filename}.sav'
    pickle._dump(model, open(filename_model, 'wb'))

    print("Predicting Data...")
    tst_strt = time.time()
    prediction = model.predict(x_test)
    tst_end = time.time()
    print("Logistic Regression Test Time", str(round(tst_end - tst_strt, 2)) + " sec")
    accuracy = round(model.score(x_test, y_test) * 100, 2)
    print("Logistic Regression Accuracy", accuracy, " %")
    print("Mean Error", metrics.mean_squared_error(np.asarray(y_test), prediction))
    print("Confusion Matrix: \n", confusion_matrix(y_test, prediction))


def decision_tree(x_tr, x_ts, y_tr, y_ts, filename='decision_tree'):
    print("\nDecision Tree")
    x_train = x_tr
    x_test = x_ts
    y_train = y_tr
    y_test = y_ts

    # Model for GridSearchCV To Search for best Max Depth
    tree = DecisionTreeClassifier(criterion='entropy', random_state=100, min_samples_leaf=100)

    print("Calculating Best HyperParameter...")
    trt_strt = time.time()
    best_params = decision_tree_hype(tree, x_train, y_train)
    max_depth = best_params['max_depth']
    #max_depth = 10
    print("Best max depth: ", max_depth)

    print("Fitting Data...")
    # Entropy (information gain) : Select best attribute.
    tree = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=max_depth, min_samples_leaf=100)
    tree.fit(x_train, y_train)
    trt_end = time.time()
    print("Decision Tree Train Time", str(round(trt_end - trt_strt, 2)) + " sec")

    with open(f'Trained Models\Models_Train_Time.txt', 'a') as the_file:
        the_file.write(f'Decision Tree Train Time: {str(round(trt_end - trt_strt, 2))} Seconds\n')

    # Save The Model into File
    filename_model = f'Trained Models\{filename}.sav'
    pickle._dump(tree, open(filename_model, 'wb'))

    print("Predicting Data...")
    tst_strt = time.time()
    prediction = tree.predict(x_test)
    tst_end = time.time()
    print("Decision Tree Test Time", str(round(tst_end - tst_strt, 2)) + " sec")

    accuracy = round(tree.score(x_test, y_test) * 100, 2)
    print("Decision Tree Accuracy", str(accuracy) + ' %')
    print("Decision Tree MSE", metrics.mean_squared_error(np.asarray(y_test), prediction))
    print("Confusion Matrix: \n", confusion_matrix(y_test, prediction))


def knn(x_tr, x_ts, y_tr, y_ts, filename='knn'):
    print("\nK-NN")
    x_train = x_tr
    x_test = x_ts
    y_train = y_tr
    y_test = y_ts

    k_nn = KNeighborsClassifier()

    print("Calculating Best HyperParameter...")
    trt_strt = time.time()
    best_params = knn_hype(k_nn, x_train, y_train)
    k = best_params['n_neighbors']
    #k = 15
    print("Best KNeighbors: ", k)

    print("Fitting Data...")
    k_nn = KNeighborsClassifier(n_neighbors=k)
    k_nn.fit(x_train, y_train)
    trt_end = time.time()
    print("K-NN Train Time", str(round(trt_end - trt_strt, 2)) + " sec")

    with open('Trained Models\Models_Train_Time.txt', 'a') as the_file:
        the_file.write(f'K NN Train Time: {str(round(trt_end - trt_strt, 2))} Seconds\n')

    # Save The Model into File
    filename_model = f'Trained Models\{filename}.sav'
    pickle._dump(k_nn, open(filename_model, 'wb'))

    print("Predicting Data...")
    tst_strt = time.time()
    prediction = k_nn.predict(x_test)
    tst_end = time.time()
    print("K-NN Test Time", str(round(tst_end - tst_strt, 2)) + " sec")

    accuracy = round(k_nn.score(x_test, y_test) * 100, 2)
    print("K-NN Accuracy", str(accuracy) + ' %')
    print("K-NN MSE", metrics.mean_squared_error(np.asarray(y_test), prediction))
    print("Confusion Matrix: \n", confusion_matrix(y_test, prediction))


def random_forest(x_tr, x_ts, y_tr, y_ts, filename='random_forest'):
    print("\nRandom Forest")
    x_train = x_tr
    x_test = x_ts
    y_train = y_tr
    y_test = y_ts

    rnd_frst = RandomForestClassifier(criterion='entropy', random_state=100, max_depth=10, min_samples_leaf=100)

    print("Calculating Best HyperParameter...")
    trt_strt = time.time()
    best_params = random_forest_hype(rnd_frst, x_train, y_train)
    estms = best_params['n_estimators']
    #estms = 1000
    print("Best n_estimators: ", estms)

    print("Fitting Data...")
    rnd_frst = RandomForestClassifier(n_estimators=estms, criterion='entropy', random_state=100, max_depth=10, min_samples_leaf=100)
    rnd_frst.fit(x_train, y_train)
    trt_end = time.time()
    print("Random Forest Train Time", str(round(trt_end - trt_strt, 2)) + " sec")

    with open('Trained Models\Models_Train_Time.txt', 'a') as the_file:
        the_file.write(f'Random Forest Train Time: {str(round(trt_end - trt_strt, 2))} Seconds\n')

    # Save The Model into File
    filename_model = f'Trained Models\{filename}.sav'
    pickle._dump(rnd_frst, open(filename_model, 'wb'))

    print("Predicting Data...")
    tst_strt = time.time()
    prediction = rnd_frst.predict(x_test)
    tst_end = time.time()
    print("Random Forest Test Time", str(round(tst_end - tst_strt, 2)) + " sec")

    accuracy = round(rnd_frst.score(x_test, y_test) * 100, 2)
    print("Random Forest Accuracy", str(accuracy) + ' %')
    print("Random Forest MSE", metrics.mean_squared_error(np.asarray(y_test), prediction))
    print("Confusion Matrix: \n", confusion_matrix(y_test, prediction))


def svm_one_vs_one_class(x_tr, x_ts, y_tr, y_ts, filename='svm_one_vs_one_class'):
    print("\n\nSVM One vs One Classification")
    x_train = x_tr
    x_test = x_ts
    y_train = y_tr
    y_test = y_ts

    print("Fitting Data...")
    trt_strt = time.time()
    # Kernal: linear , poly, rbf, sigmoid
    svm_model_linear_ovo = SVC(kernel='rbf', gamma='auto', C=10, decision_function_shape='ovo').fit(x_train, y_train)
    trt_end = time.time()
    print("SVM OneVsOne Train Time", str(round(trt_end - trt_strt, 2)) + " sec")
    ''' 'auto' value of gamma means gamma = 1 / n_feats,
    while 'scale' value of gamma means gamma = 1 / (n_feats * X.var()) but this takes much time'''

    with open(f'Trained Models\Models_Train_Time.txt', 'a') as the_file:
        the_file.write(f'SVM OneVsOne Train Time: {str(round(trt_end - trt_strt, 2))} Seconds\n')

    # Save The Model into File
    filename_model = f'Trained Models\{filename}.sav'
    pickle._dump(svm_model_linear_ovo, open(filename_model, 'wb'))

    print("Predicting Data...")
    tst_strt = time.time()
    prediction = svm_model_linear_ovo.predict(x_test)
    tst_end = time.time()
    print("SVM OneVsRest Test Time", str(round(tst_end - tst_strt, 2)) + " sec")

    accuracy = svm_model_linear_ovo.score(x_test, y_test)
    print("SVM OneVsOne Accuracy", str(accuracy * 100) + ' %')
    print("SVM OneVsOne MSE", metrics.mean_squared_error(np.asarray(y_test), prediction))


def svm_one_vs_rest_class(x_tr, x_ts, y_tr, y_ts, filename='svm_one_vs_rest_class'):
    print("\n\nSVM One vs Rest Classification")
    x_train = x_tr
    x_test = x_ts
    y_train = y_tr
    y_test = y_ts

    print("Fitting Data...")
    trt_strt = time.time()
    #svm_model_linear_ovr = SVC(kernel='rbf', gamma='auto', C=10,  decision_function_shape='ovr').fit(x_train, y_train)
    svm_model_linear_ovr = LinearSVC(multi_class='ovr', C=1)
    svm_model_linear_ovr.fit(x_train, y_train)
    trt_end = time.time()
    print("SVM OneVsRest Train Time", str(round(trt_end - trt_strt, 2)) + " sec")

    with open('Trained Models\Models_Train_Time.txt', 'a') as the_file:
        the_file.write(f'SVM OneVsRest Train Time: {str(round(trt_end - trt_strt, 2))} Seconds\n')

    # Save The Model into File
    filename_model = f'Trained Models\{filename}.sav'
    pickle._dump(svm_model_linear_ovr, open(filename_model, 'wb'))

    print("Predicting Data...")
    tst_strt = time.time()
    prediction = svm_model_linear_ovr.predict(x_test)
    tst_end = time.time()
    print("SVM OneVsRest Test Time", str(round(tst_end - tst_strt, 2)) + " sec")

    accuracy = svm_model_linear_ovr.score(x_test, y_test)
    print("SVM OneVsRest Accuracy", str(accuracy * 100) + ' %')
    print("SVM OneVsRest MSE", metrics.mean_squared_error(np.asarray(y_test), prediction))


# ------------------------------------------------------------------------------------------
# Hyper Parameters
def decision_tree_hype(model, x_train, y_train):
    parameters = {'max_depth': [1, 5, 10, 100, 1000]}
    tun = GridSearchCV(model, parameters, cv=5) # cv = Cross Validation (Splitter)
    tun.fit(x_train, y_train)
    return tun.best_params_

def knn_hype(model, x_train, y_train):
    parameters = {'n_neighbors': [1, 5, 10, 15]}
    tun = GridSearchCV(model, parameters, cv=5)
    tun.fit(x_train, y_train)
    return tun.best_params_

def random_forest_hype(model, x_train, y_train):
    parameters = {'n_estimators': [10, 150, 1000, 1500]}
    tun = GridSearchCV(model, parameters, cv=5)
    tun.fit(x_train, y_train)
    return tun.best_params_

def svm_ovr_hype(model, x_train, y_train):
    print("Begin finding the best C")
    parameters = {'C': [2**-5, 2**-2, 2**5]}
    tun = GridSearchCV(model, parameters, cv=5)
    print("Fitting..AfterSearch")
    tun.fit(x_train, y_train)
    return tun.best_params_, tun

# ------------------------------------------------------------------------------------------
# Testing By New Samples
def ready_logistic_regression(x_ts, y_ts, filename='logistic_regression'):
    print("\n\n(Ready) Logistic Regression")
    x_test = x_ts
    y_test = y_ts

    # Load The Model from File
    filename_model = f'Trained Models\{filename}.sav'
    model = pickle.load(open(filename_model, 'rb'))

    print("Predicting Data...")
    tst_strt = time.time()
    prediction = model.predict(x_test)
    tst_end = time.time()

    with open('Trained Models\Models_Test_Time.txt', 'a') as the_file:
        the_file.write(f'Logistic Regression Test Time: {str(round(tst_end - tst_strt, 2))} Seconds\n')

    print("Logistic Regression Test Time", str(round(tst_end - tst_strt, 2)) + " sec")
    accuracy = round(model.score(x_test, y_test) * 100, 2)
    print("Logistic Regression Accuracy", accuracy, " %")
    print("Mean Error", metrics.mean_squared_error(np.asarray(y_test), prediction))
    print("Confusion Matrix: \n", confusion_matrix(y_test, prediction))

    with open('Trained Models\Models_Acc_Time.txt', 'a') as the_file:
        the_file.write(f'Logistic Regression Accuracy: {str(accuracy)} %\n')

def ready_decision_tree(x_ts, y_ts, filename='decision_tree'):
    print("\n\n(Ready) Decision Tree")
    x_test = x_ts
    y_test = y_ts

    # Load The Model from File
    filename_model = f'Trained Models\{filename}.sav'
    tree = pickle.load(open(filename_model, 'rb'))

    print("Predicting Data...")
    tst_strt = time.time()
    prediction = tree.predict(x_test)
    tst_end = time.time()
    print("Decision Tree Test Time", str(round(tst_end - tst_strt, 2)) + " sec")

    with open('Trained Models\Models_Test_Time.txt', 'a') as the_file:
        the_file.write(f'Decision Tree Test Time: {str(round(tst_end - tst_strt, 2))} Seconds\n')

    accuarcy = round(tree.score(x_test, y_test) * 100, 2)
    print("Decision Tree Accuracy", str(accuarcy) + ' %')
    print("Decision Tree MSE", metrics.mean_squared_error(np.asarray(y_test), prediction))
    print("Confusion Matrix: \n", confusion_matrix(y_test, prediction))

    with open('Trained Models\Models_Acc_Time.txt', 'a') as the_file:
        the_file.write(f'Decision Tree Accuracy: {str(accuarcy)} %\n')

def ready_knn(x_ts, y_ts, filename='knn'):
    print("\n(Ready) K-NN")
    x_test = x_ts
    y_test = y_ts

    # Load The Model from File
    filename_model = f'Trained Models\{filename}.sav'
    k_nn = pickle.load(open(filename_model, 'rb'))

    print("Predicting Data...")
    tst_strt = time.time()
    prediction = k_nn.predict(x_test)
    tst_end = time.time()
    print("K-NN Test Time", str(round(tst_end - tst_strt, 2)) + " sec")

    with open('Trained Models\Models_Test_Time.txt', 'a') as the_file:
        the_file.write(f'K NN Test Time: {str(round(tst_end - tst_strt, 2))} Seconds\n')

    accuracy = round(k_nn.score(x_test, y_test) * 100, 2)
    print("K-NN Accuracy", str(accuracy) + ' %')
    print("K-NN MSE", metrics.mean_squared_error(np.asarray(y_test), prediction))
    print("Confusion Matrix: \n", confusion_matrix(y_test, prediction))

    with open('Trained Models\Models_Acc_Time.txt', 'a') as the_file:
        the_file.write(f'K NN Accuracy: {str(accuracy)} %\n')

def ready_random_forest(x_ts, y_ts, filename='random_forest'):
    print("\n(Ready) ready_random_forest")
    x_test = x_ts
    y_test = y_ts

    # Load The Model from File
    filename_model = f'Trained Models\{filename}.sav'
    rnd_frst = pickle.load(open(filename_model, 'rb'))

    print("Predicting Data...")
    tst_strt = time.time()
    prediction = rnd_frst.predict(x_test)
    tst_end = time.time()
    print("Random Forest Test Time", str(round(tst_end - tst_strt, 2)) + " sec")

    with open('Trained Models\Models_Test_Time.txt', 'a') as the_file:
        the_file.write(f'Random Forest Test Time: {str(round(tst_end - tst_strt, 2))} Seconds\n')

    accuracy = round(rnd_frst.score(x_test, y_test) * 100, 2)
    print("Random Forest Accuracy", str(accuracy) + ' %')
    print("Random Forest MSE", metrics.mean_squared_error(np.asarray(y_test), prediction))
    print("Confusion Matrix: \n", confusion_matrix(y_test, prediction))

    with open('Trained Models\Models_Acc_Time.txt', 'a') as the_file:
        the_file.write(f'Random Forest Accuracy: {str(accuracy)} %\n')

def ready_svm_one_vs_one_class(x_ts, y_ts, filename='svm_one_vs_one_class'):
    print("\n\n(Ready) SVM One vs One Classification")
    x_test = x_ts
    y_test = y_ts

    # Load The Model from File
    filename_model = f'Trained Models\{filename}.sav'
    svm_model_linear_ovo = pickle.load(open(filename_model, 'rb'))

    print("Predicting Data...")
    tst_strt = time.time()
    prediction = svm_model_linear_ovo.predict(x_test)
    tst_end = time.time()
    print("SVM OneVsOne Test Time", str(round(tst_end - tst_strt, 2)) + " sec")

    with open('Trained Models\Models_Test_Time.txt', 'a') as the_file:
        the_file.write(f'SVM OneVsOne Test Time: {str(round(tst_end - tst_strt, 2))} Seconds\n')

    accuracy = round(svm_model_linear_ovo.score(x_test, y_test) * 100, 2)
    print("SVM OneVsOne Accuracy", str(accuracy) + ' %')
    print("SVM OneVsOne MSE", metrics.mean_squared_error(np.asarray(y_test), prediction))
    print("Confusion Matrix: \n", confusion_matrix(y_test, prediction))

    with open('Trained Models\Models_Acc_Time.txt', 'a') as the_file:
        the_file.write(f'SVM OneVsOne Accuracy: {str(accuracy)} %\n')

def ready_svm_one_vs_rest_class(x_ts, y_ts, filename='svm_one_vs_rest_class'):
    print("\n\n(Ready) SVM One vs Rest Classification")
    x_test = x_ts
    y_test = y_ts

    # Load The Model from File
    filename_model = f'Trained Models\{filename}.sav'
    svm_model_linear_ovr = pickle.load(open(filename_model, 'rb'))

    print("Predicting Data...")
    tst_strt = time.time()
    prediction = svm_model_linear_ovr.predict(x_test)
    tst_end = time.time()
    print("SVM OneVsRest Test Time", str(round(tst_end - tst_strt, 2)) + " sec")

    with open('Trained Models\Models_Test_Time.txt', 'a') as the_file:
        the_file.write(f'SVM OneVsRest Test Time: {str(round(tst_end - tst_strt, 2))} Seconds\n')

    accuracy = round(svm_model_linear_ovr.score(x_test, y_test) * 100, 2)
    print("SVM OneVsRest Accuracy", str(accuracy) + ' %')
    print("SVM OneVsRest MSE", metrics.mean_squared_error(np.asarray(y_test), prediction))
    print("Confusion Matrix: \n", confusion_matrix(y_test, prediction))

    with open('Trained Models\Models_Acc_Time.txt', 'a') as the_file:
        the_file.write(f'SVM OneVsRest Accuracy: {str(accuracy)} %\n')


