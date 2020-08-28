import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_cumsum_var_pca(pca):
    #Plotting the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('Hotel Dataset Explained Variance')
    plt.show()


def show_bar_graph(filename):
    with open(f'Trained Models\{filename}.txt', 'r') as the_file:
            models = the_file.read().splitlines()
            print(len(models))
            models_sec = []
            models_names = []
            for i in range(len(models)):
                models_sec.append(float(models[i].split(': ')[1].split(' ')[0]))
                models_names.append(f"{models[i].split(': ')[0].split(' ')[0]} {models[i].split(': ')[0].split(' ')[1]}")

            print(models_sec)
            print(models_names)

            if 'Train' in filename:
                title = 'Training Time'
                y_label = 'Seconds'
            if 'Test' in filename:
                title = 'Testing Time'
                y_label = 'Seconds'
            if 'Acc' in filename:
                title = 'Accuracy (Score)'
                y_label = 'Accuracy Percentage'

            y_pos = np.arange(len(models_names))
            performance = models_sec
            plt.bar(y_pos, performance, align='center', alpha=0.6)
            plt.xticks(y_pos, models_names)
            plt.ylabel(y_label)
            plt.title(title)
            plt.show()


#show_bar_graph('Models_Train_Time')
#show_bar_graph('Models_Test_Time')
#show_bar_graph('Models_Acc_Time')