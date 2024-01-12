import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import COMP0078Q1

def split_data(data,prop,seed):
    np.random.seed(seed)
    training = data.sample(frac=prop)
    testing = data[~data.index.isin(training.index)]
    return training,testing

def regression_mse(w,data_x,data_y):
    mse = sum(np.power(data_y - np.matmul(data_x,w),2))/len(data_x)
    return mse

def baseline_regresion(training,testing):
    baseline_train = np.ones(len(training)).reshape((-1,1))
    baseline_test = np.ones(len(testing)).reshape((-1,1))
    y_train = training["MEDV"]
    y_test = testing["MEDV"]
    w_train = COMP0078Q1.regression_coef(baseline_train,y_train)
    
    mse_train = regression_mse(w_train,baseline_train,y_train)
    mse_test = regression_mse(w_train,baseline_test,y_test)

    return mse_train,mse_test


def single_attr_regression(training,testing,indx):
    x_train = np.array([training.iloc[:,indx],np.ones(len(training))]).T
    y_train = training["MEDV"]
    x_test = np.array([testing.iloc[:,indx],np.ones(len(testing))]).T
    y_test = testing["MEDV"]

    w_train = COMP0078Q1.regression_coef(x_train,y_train)

    mse_train = regression_mse(w_train,x_train,y_train)
    mse_test = regression_mse(w_train,x_test,y_test)
    
    return mse_train,mse_test


def all_attr_regression(training,testing):
    x_train_temp = training.iloc[:,:12]
    x_train_temp["bias"] = 1
    x_train = np.array(x_train_temp)
    y_train = training["MEDV"]
    
    x_test_temp = testing.iloc[:,:12]
    x_test_temp["bias"] = 1
    x_test = np.array(x_test_temp)
    y_test= testing["MEDV"]

    w_train = COMP0078Q1.regression_coef(x_train,y_train)

    mse_train = regression_mse(w_train,x_train,y_train)
    mse_test = regression_mse(w_train,x_test,y_test)

    return mse_train,mse_test

if __name__ == "__main__":
    data = pd.read_csv("Boston-filtered.csv")
    
    MSE_train_a = []
    MSE_test_a = []
    MSE_train_b = []
    MSE_test_b = []
    MSE_train_c = []
    MSE_test_c = []

    run = 20

    for i in range(run):
        training,testing = split_data(data,2/3,seed=i)
        
        # (a) baseline regression
        mse_train_a,mse_test_a = baseline_regresion(training,testing)
        MSE_train_a.append(mse_train_a)
        MSE_test_a.append(mse_test_a)


        # (b) single attribute regression plus a bias term
        MSEtrain = []
        MSEtest = []
        for indx in range(12):
            mse_train_b,mse_test_b = single_attr_regression(training,testing,indx)
            MSEtrain.append(mse_train_b)
            MSEtest.append(mse_test_b)
        MSE_train_b.append(MSEtrain)
        MSE_test_b.append(MSEtest)

        # (c) all attributes regression plus a bias term
        mse_train_c,mse_test_c = all_attr_regression(training,testing)
        MSE_train_c.append(mse_train_c)
        MSE_test_c.append(mse_test_c)
    
    result_train_a = sum(MSE_train_a)/run
    result_test_a = sum(MSE_test_a)/run
    result_train_b = np.sum(np.array(MSE_train_b),axis=0)/run
    result_test_b = np.sum(np.array(MSE_test_b),axis=0)/run
    result_train_c = sum(MSE_train_c)/run
    result_test_c = sum(MSE_test_c)/run

    print("results for (a)")
    print(result_train_a)
    print(result_test_a)
    print("results for (b)")
    print(result_train_b)
    print(result_test_b)
    print("results for (c)")
    print(result_train_c)
    print(result_test_c)

