import statistics
import pandas as pd
import matplotlib.pyplot as plt

def Standardize(list):
    average = statistics.mean(list)
    standard_deviation = statistics.stdev(list)
    return [(element-average)/standard_deviation for element in list]

def Error(m,b,data):
    error=0
    for i in range(data["Num of Samples"]):
        error += (data["Final Marks"]["Values"][i] - (m * data["Midterm Marks"]["Values"][i] + b)) ** 2
    return error/data["Num of Samples"]

def GPD(m,b,data,parameter):
    if (parameter != 'm' and parameter != 'b'):
        return None
    diff=0
    for i in range(data["Num of Samples"]):
        sum_term = -1*(data["Final Marks"]["Values"][i] - (m*data["Midterm Marks"]["Values"][i] + b))
        if parameter == 'm':
            sum_term *= data["Midterm Marks"]["Values"][i]
        diff += sum_term
    return 2*diff/data["Num of Samples"]

def update(m,b,data,parameter,alpha):
    if (parameter == 'm'):
        return m-alpha*GPD(m,b,data,'m')
    if (parameter == 'b'):
        return b-alpha*GPD(m,b,data,'b')
    return None

def showmodel(m,b,data,title):
    plt.scatter(data["Midterm Marks"]["Values"], data["Final Marks"]["Values"], color='#00008B',label='Normalized Data Set')
    plt.plot(data["Midterm Marks"]["Values"], [m * value + b for value in data["Midterm Marks"]["Values"]],color='#FF0000', label='Linear Regression Model')
    plt.xlabel("Normalized Midterm Marks")
    plt.ylabel("Normalized Final Marks")
    plt.legend(loc="lower right")
    plt.title(title)
    plt.show()

def showerror(error,title):
    plt.plot(range(1,len(error)+1), error,color='#FF0000')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Error")
    plt.title(title)
    plt.show()

def trainModel(m,b,alpha,data,num_of_iterations):
    error=[]
    for i in range(num_of_iterations):
        m_old = m
        b_old = b
        m = update(m_old, b_old, data, 'm', alpha)
        b = update(m_old, b_old, data, 'b', alpha)
        error.append(Error(m,b,data))
    showmodel(m, b, data, "Model After {} iterations(m={:.3f}, b={:.3e})".format(num_of_iterations,m,b))
    showerror(error,"Error Through {} iterations".format(num_of_iterations))

raw_data = pd.read_csv(r'https://raw.githubusercontent.com/tofighi/MachineLearning/master/datasets/student_marks.csv')
data={
    "Midterm Marks": {
        "Values" : Standardize(raw_data['Midterm mark']),
        "Standard Deviation" : statistics.stdev(raw_data['Midterm mark']),
        "Average" : statistics.mean(raw_data['Midterm mark'])},
    "Final Marks": {
        "Values": Standardize(raw_data['Final mark']),
        "Standard Deviation": statistics.stdev(raw_data['Final mark']),
        "Average": statistics.mean(raw_data['Final mark'])},
    "Num of Samples": raw_data.__len__()
}
m=-0.5
b=0
alpha=0.00001
showmodel(m,b,data,"Initial Model Without Training")
trainModel(m,b,alpha,data,100)
trainModel(m,b,alpha,data,2000)
trainModel(m,b,alpha,data,10000)



