import statistics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math #math.log is faster then np.log as np can work with lists. I dont use that fuctionality here though
import warnings  # use to ignore warnings from pyplot
import time  # use to measure runtime of code
from multiprocessing import Pool #use to speed up model training, by running them concurently
from sklearn.linear_model import LogisticRegression


# Standardize the parameter list
def Standardize(list):
    average = statistics.mean(list)
    standard_deviation = statistics.stdev(list)
    return [(element - average) / standard_deviation for element in list]

#Split the training data l into batches of size n
def Split(l,n):
    return [l[i:i + n] for i in range(0, len(l), n)]

# Compute the output of the model
def Model(thetas, inputs):
    z = 0
#    for i in range(0, len(inputs)):
#        z += inputs[i] * thetas[i]
    z = np.sum(thetas*inputs)   #element-wise multiplication is faster than my implementation above
    return 1 / (1 + np.e**(-1 * z))

#Calculate error of model
def Error(thetas, features, targets):
    error = 0
    for i in range(len(features)):
        error += targets[i] * math.log(Model(thetas, features[i])) + (1 - targets[i]) * (math.log(1 - Model(thetas, features[i])))
    # print("Error is: {}".format(-1*error/len(features)))
    return -1 * error / len(features)

#use Gradient Partial Descent algorithm to calculate the updated model parameters 'thetas'
def GPD_Iterate(thetas, features, targets, alpha):
    new_thetas = list()
    for j in range(len(thetas)):
        sum_term = 0
        for i in range(len(features)):
            sum_term += features[i][j] * (targets[i] - Model(thetas, features[i]))
        sum_term = sum_term / len(features)
        new_thetas.append(thetas[j] + alpha * sum_term)
    return new_thetas

#plot the error of the model for each iteration, using a list which contains the error values
def showerror(error, title):
    plt.plot(range(1, len(error) + 1), error, color='#FF0000')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Error")
    plt.title(title)
    plt.show()

#function to automate the training of a model.
def trainModel(thetas, features, targets, alpha, num_of_epochs, batch_size):
    error = [] #keep track of the error values for each iteration
    #create a list of batches. if batch size is equal to 0, list will only have 1 element.
    #convert the pandas dataframe into a numpy array. Results in ~15X faster training
    if (batch_size == 0):
        features_batches = [features.to_numpy()]
        targets_batches = [targets.to_numpy()]
        batch_size = len(features.index)
    else:
        features_batches = Split(features, batch_size)
        targets_batches = Split(targets, batch_size)
        features_batches = [batch.to_numpy() for batch in features_batches]
        targets_batches = [batch.to_numpy() for batch in targets_batches]
    features = features.to_numpy()
    targets = targets.to_numpy()

    for i in range(num_of_epochs):
        for j in range(len(features_batches)):
            thetas = GPD_Iterate(thetas, features_batches[j], targets_batches[j], alpha)
        error.append(Error(thetas, features, targets))
        if (i == 500 or i == 1000 or i == 10000 or i == 100000):
            showerror(error, "Error Through {} epochs (alpha = {}, batch_size = {})".format(i, alpha, batch_size))
        if (i % 1000 == 0):
            print("Model with alpha of {}, num of epochs of {}, batch_size of {} Currently on Epoch: {}".format(alpha, num_of_epochs, batch_size, i))
    showerror(error, "Error Through {} epochs (alpha = {}, batch_size = {})".format(num_of_epochs, alpha, batch_size))
    print("Model with alpha of {}, num of epochs of {}, batch_size of {} has the following thetas:{}".format(alpha, num_of_epochs, batch_size, thetas))

#Main Function
if __name__ == '__main__':
    start_time = time.time() #used to calculate runtime
    pd.set_option('display.max_columns', None) #dont shorten pandas outputs
    pd.set_option('display.max_rows', None) #dont shorten pandas outputs

    raw_data = pd.read_csv(r'https://raw.githubusercontent.com/tofighi/MachineLearning/master/datasets/heart.csv')

    # EDA analysis:
    # How much is the percentage of each class 0 and 1
    print("The values in chd are as follows:")
    print(raw_data['chd'].value_counts())

    # How many missing values do we have
    print("\nThe missing values are as follows:")
    print(raw_data.isna().sum())

    # How many categorical variables do you have in your features
    print("\nThe column raw_data types are as follows:")
    print(raw_data.info())

    # What features have the maximum corrolation
    f = plt.figure(figsize=(19, 15))
    plt.matshow(raw_data.corr(), fignum=f.number)
    plt.xticks(range(raw_data.shape[1]), raw_data.columns, fontsize=14, rotation=45)
    plt.yticks(range(raw_data.shape[1]), raw_data.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        plt.show()

    # Encode the categorical variable(s)
    one_hot = pd.get_dummies(raw_data['famhist'])
    raw_data = raw_data.drop('famhist', 'columns')
    raw_data = raw_data.join(one_hot)
    print("\nEncoded \'famhist\' using one-hot encoding:")
    print(raw_data.head())

    # Create new Data Frames for feature data, and target data
    Feature_Data = raw_data.iloc[:, np.r_[1:9, 10, 11]].copy()
    Feature_Data.columns = list(Feature_Data.columns)[:-2] + ['famhist_true', 'famhist_false']  # Rename last 2 columns. Not nessessary, but nice to have
    Target_Data = raw_data.iloc[:, 9].copy()

    # Standardize Feature_Data columns, except one-hot-encoded columns
    for column in Feature_Data.columns[:-2]:
        Feature_Data.loc[:, column] = Standardize(Feature_Data.loc[:, column])
    Feature_Data.insert(0, 'x0', 1) #add a column to represent 'x0' to simplify model training functions above. x0 is always 1

    # Initialize The Model
    np.random.seed(70)  # option for reproducibility
    thetas = np.random.rand(len(Feature_Data.columns))

    #Train the models. Use multiprocessing, as training takes ~2.5 hours if running sequencially.
    print("Start Training using Multiprocessing:")
    pool = Pool(4)  #Define 4 workers in the pool. If your CPU does not have 4 cores, or you want to use less than 4 cores, reduce this value
    pool.starmap(trainModel, [(thetas, Feature_Data, Target_Data, 0.0001, 25000, 0),
                              (thetas, Feature_Data, Target_Data, 0.0001, 10000, 50),
                              (thetas, Feature_Data, Target_Data, 0.001, 25000, 0),
                              (thetas, Feature_Data, Target_Data, 0.001, 10000, 50)]) # Add commands to the pool. each worker will call the 'trainModel' function with an element from the list
    pool.close() #wait for all workers to finish
    pool.join() #combine all threads, so any line after this does not use multiple cores

    #Train model using sklearn
    LogisticRegression_Model = LogisticRegression().fit(Feature_Data,Target_Data)
    print("Model trained with sklearn has the following thetas:")
    print(np.hstack((LogisticRegression_Model.intercept_[:,None], LogisticRegression_Model.coef_)))


    print("Run Time with mp is: {} Seconds".format(time.time() - start_time))
