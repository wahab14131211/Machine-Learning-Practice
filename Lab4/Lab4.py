import statistics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math #math.log is faster then np.log as np can work with lists. I dont use that fuctionality here though
import warnings  # use to ignore warnings from pyplot
import time  # use to measure runtime of code
from multiprocessing import Pool #use to speed up model training, by running them concurently

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Main Function
if __name__ == '__main__':
    start_time = time.time() #used to calculate runtime
    pd.set_option('display.max_columns', None) #dont shorten pandas outputs
    pd.set_option('display.max_rows', None) #dont shorten pandas outputs

    raw_data = pd.read_csv(r'https://raw.githubusercontent.com/tofighi/MachineLearning/master/datasets/heart.csv')

    # Encode the categorical variable(s)
    one_hot = pd.get_dummies(raw_data['famhist'])
    raw_data = raw_data.drop('famhist', 'columns')
    raw_data = raw_data.join(one_hot)

    #extract features and labels from raw_data, and split into train and test batches
    # Create new Data Frames for feature data, and target data
    #Feature_Data = raw_data.iloc[:, np.r_[1:9, 10, 11]].copy()
    #Feature_Data.columns = list(Feature_Data.columns)[:-2] + ['famhist_true', 'famhist_false']  # Rename last 2 columns. Not nessessary, but nice to have
    #Target_Data = raw_data.iloc[:, 9].copy()
    #features_train,features_test,labels_train,labels_test = train_test_split(Feature_Data, Target_Data, random_state=43)

    ### UNCOMMENT THIS SECTION TO ONLY HAVE 1 COLUMN FOR 1 HOT ENCODING ###
    Feature_Data = raw_data.iloc[:, np.r_[1:9, 10]].copy()
    Feature_Data.columns = list(Feature_Data.columns)[:-1] + ['famhist_true']  # Rename last 2 columns. Not nessessary, but nice to have
    Target_Data = raw_data.iloc[:, 9].copy()
    features_train,features_test,labels_train,labels_test = train_test_split(Feature_Data, Target_Data, random_state=43)

    #normalize features
    Standardizer = StandardScaler()
    features_train = Standardizer.fit_transform(features_train)
    features_test = Standardizer.transform(features_test)

    #Apply basic logistic regression
    logistic_regression_model = LogisticRegression().fit(features_train,labels_train)
    print("Accuracy with basic logistic regression model is: {}".format(logistic_regression_model.score(features_test,labels_test)))

    #Calculate PCA and then use logistic regression
    pca = PCA()
    features_train_pca = pca.fit_transform(features_train)
    features_test_pca = pca.transform(features_test)
    logistic_regression_model_pca = LogisticRegression().fit(features_train_pca, labels_train)
    print("Accuracy for logistic regression model with all PCs is: {}".format(logistic_regression_model_pca.score(features_test_pca, labels_test)))

    #Calculate PCs which account for 90% varience and then use logistic regression
    pca = PCA(0.9)
    features_train_pca_90 = pca.fit_transform(features_train)
    print("The number of PCs that explain 90% of the variation is: {}".format(features_train_pca_90.shape[1]))
    print("The first 2 PCs for logistic regression explains {}% of the variation".format(100*(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])))
    features_test_pca_90 = PCA(n_components=features_train_pca_90.shape[1]).fit_transform(features_test) #had an issue where PCA(0.9) selected a different number of components for train set and test set. this line will force the number of PCs in the test set to match the train set
    logistic_regression_model_pca_90 = LogisticRegression().fit(features_train_pca_90, labels_train)
    print("Accuracy for logistic regression model with PCs resulting in 90% of variance is: {}".format(logistic_regression_model_pca_90.score(features_test_pca_90, labels_test)))


    #K means clustering
    features_train = np.concatenate((features_train,features_test))
    labels_train = pd.concat([labels_train,labels_test])
    features_test = features_train
    labels_test = labels_train

    k_means_model = KMeans(n_clusters=2, random_state=43).fit(features_train)
    test_samples = len(labels_test)
    predictions = k_means_model.predict(features_test)
    test_accuracy_1 = sum([1 for i in range(test_samples) if labels_test.to_numpy()[i] == predictions[i]])/test_samples #Assume default labels are correct
    test_accuracy_2 = sum([1 for i in range(test_samples) if labels_test.to_numpy()[i] != predictions[i]])/test_samples #Assume default labels are flipped
    test_accuracy = max(test_accuracy_1,test_accuracy_2)
    print("Accuracy for basic k means clustering is: {} ".format(test_accuracy))


    #use kmeans with 2 pcs
    pca = PCA(n_components=2)
    features_train_2pc = pca.fit_transform(features_train)
    features_test_2pc = pca.transform(features_test)
    k_means_model_2pc = KMeans(n_clusters=2,random_state=43).fit(features_train_2pc)
    predictions = k_means_model_2pc.predict(features_test_2pc)
    test_accuracy_1 = sum([1 for i in range(test_samples) if labels_test.to_numpy()[i] == predictions[i]])/test_samples #Assume cluster 1 represents class 1
    test_accuracy_2 = sum([1 for i in range(test_samples) if labels_test.to_numpy()[i] != predictions[i]])/test_samples #Assume cluster 1 represents class 2
    test_accuracy = max(test_accuracy_1,test_accuracy_2)
    print("Accuracy for 2pc k means clustering is: {}".format(test_accuracy))



    # plotting clusters
    plt.scatter(features_test_2pc[predictions == 0, 0], features_test_2pc[predictions == 0, 1], c='blue', edgecolor='black', label='cluster 1')
    plt.scatter(features_test_2pc[predictions == 1, 0], features_test_2pc[predictions == 1, 1], c='red', edgecolor='black', label='cluster 2')
    plt.title("K-Means Clusters showing clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()

    # plotting classes
    plt.scatter(features_test_2pc[labels_test.to_numpy() == 0, 0], features_test_2pc[labels_test.to_numpy() == 0, 1], c='blue', edgecolor='black', label='chd not present')
    plt.scatter(features_test_2pc[labels_test.to_numpy() == 1, 0], features_test_2pc[labels_test.to_numpy() == 1, 1], c='red', edgecolor='black', label='chd present')
    plt.title("K-Means Clusters showing labels")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()
