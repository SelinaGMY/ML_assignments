import numpy as np
from matplotlib import pyplot as plt
import COMP0078Q1
import COMP0078Q2

def transform_data_x_newbasis(data_x,k):
    trans_data = np.ones((len(data_x),k))
    row = 0
    for x in data_x:
        for col in range(k):
            trans_data[row][col] = np.sin((col+1) * np.pi * x)
        row += 1
    return trans_data

# b 
def MSE_train_newbasis(K,seed):
    np.random.seed(seed)
    x = np.random.uniform(0, 1, 30)
    y = (np.sin(2*np.pi*x))**2 + np.random.normal(0,0.07,30)

    MSE = []
    for k in K:
        trans_data = np.round(transform_data_x_newbasis(x,k),6)
        w = np.round(COMP0078Q1.regression_coef(trans_data,y),6)
        mse = sum(np.power(y-np.matmul(trans_data,w),2))/len(x)
        mse = np.round(mse,6)
        MSE.append(mse)
    
    return MSE

# c
def MSE_overfit_newbasis(K,seed):
    np.random.seed(seed)
    train_x = np.random.uniform(0, 1, 30)
    train_y = (np.sin(2*np.pi*train_x))**2 + np.random.normal(0,0.07,30)
    test_x = np.random.uniform(0, 1, 1000)
    test_y = (np.sin(2*np.pi*test_x))**2 + np.random.normal(0,0.07,1000)
    
    test_MSE = []
    for k in K:
        # training process: fit train_w
        trans_train_data = np.round(transform_data_x_newbasis(train_x,k),6)
        train_w = np.round(COMP0078Q1.regression_coef(trans_train_data,train_y),6)
        
        # testing process: apply train_w to test data
        trans_test_data = np.round(transform_data_x_newbasis(test_x,k),6)
        test_mse = sum(np.power(test_y-np.matmul(trans_test_data,train_w),2))/len(test_x)
        test_mse = np.round(test_mse,6)
        test_MSE.append(test_mse)
    
    return test_MSE

# d
def repeat_MSE_newabasis(K,iter_num):
    MSE_total = []
    test_MSE_total = []
    for i in range(iter_num):
        MSE = MSE_train_newbasis(K,seed=i)
        test_MSE = MSE_overfit_newbasis(K,seed=i)
        MSE_total.append(MSE)
        test_MSE_total.append(test_MSE)
    
    MSE_total = np.array(MSE_total)
    test_MSE_total = np.array(test_MSE_total)

    MSE_avg = np.sum(MSE_total,axis=0)/iter_num
    test_MSE_avg = np.sum(test_MSE_total,axis=0)/iter_num

    return MSE_avg,test_MSE_avg



if __name__ == "__main__":
    # b
    K = np.arange(1,19)
    MSE = MSE_train_newbasis(K, seed = 111)
    COMP0078Q2.plot_mse(K,MSE,"fig3_1.pdf","k","natural log training error")

    # c
    K = np.arange(1,19)
    test_MSE = MSE_overfit_newbasis(K, seed = 1111)
    COMP0078Q2.plot_mse(K,test_MSE,"fig3_2.pdf","k","natural log testing error")

    # d
    K = np.arange(1,19)
    iter_num = 100
    MSE_avg,test_MSE_avg = repeat_MSE_newabasis(K,iter_num)
    COMP0078Q2.plot_mse(K,MSE_avg,"fig3_3.pdf","k","average natural log training error",18)
    COMP0078Q2.plot_mse(K,test_MSE_avg,"fig3_4.pdf","k","average natural log testing error",18)

