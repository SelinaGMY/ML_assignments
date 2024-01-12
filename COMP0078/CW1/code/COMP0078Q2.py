import numpy as np
from matplotlib import pyplot as plt
import COMP0078Q1

# a(i)
def sin_grpah():
    fig,axes = plt.subplots()

    np.random.seed(1)
    x = np.random.uniform(0, 1, 30)
    y = (np.sin(2*np.pi*x))**2 + np.random.normal(0,0.07,30)
    axes.scatter(x,y,s=15)

    x1 = np.linspace(0, 1, 100)
    y1 = (np.sin(2 * np.pi * x1)) ** 2
    axes.plot(x1,y1)

    plt.savefig("fig2_1.pdf")

# a(ii)
def sin_polybase_graph(K):
    fig,axes = plt.subplots()

    np.random.seed(11)
    x = np.random.uniform(0, 1, 30)
    y = (np.sin(2*np.pi*x))**2
    axes.scatter(x,y,s=15)

    x1 = np.linspace(0, 1, 100)
    for k in K:
        trans_data = COMP0078Q1.transform_data_x(x,k)
        w = COMP0078Q1.regression_coef(trans_data,y)
        
        y1 = 0
        power = 0
        for coef in np.nditer(w):
            y1 += coef * x1 ** power
            power += 1
        axes.plot(x1,y1)
    axes.legend([f"k={i}" for i in K])
    plt.savefig("fig2_2.pdf")

# b
def MSE_train(K,seed):
    np.random.seed(seed)
    x = np.random.uniform(0, 1, 30)
    y = (np.sin(2*np.pi*x))**2 + np.random.normal(0,0.07,30)

    MSE = []
    for k in K:
        trans_data = np.round(COMP0078Q1.transform_data_x(x,k),6)
        w = np.round(COMP0078Q1.regression_coef(trans_data,y),6)
        mse = sum(np.power(y-np.matmul(trans_data,w),2))/len(x)
        mse = np.round(mse,6)
        MSE.append(mse)
    
    return MSE
    
# c
def MSE_overfit(K,seed):
    np.random.seed(seed)
    train_x = np.random.uniform(0, 1, 30)
    train_y = (np.sin(2*np.pi*train_x))**2 + np.random.normal(0,0.07,30)
    test_x = np.random.uniform(0, 1, 1000)
    test_y = (np.sin(2*np.pi*test_x))**2 + np.random.normal(0,0.07,1000)
    
    test_MSE = []
    for k in K:
        # training process: fit train_w
        trans_train_data = np.round(COMP0078Q1.transform_data_x(train_x,k),6)
        train_w = np.round(COMP0078Q1.regression_coef(trans_train_data,train_y),6)
        
        # testing process: apply train_w to test data
        trans_test_data = np.round(COMP0078Q1.transform_data_x(test_x,k),6)
        test_mse = sum(np.power(test_y-np.matmul(trans_test_data,train_w),2))/len(test_x)
        test_mse = np.round(test_mse,6)
        test_MSE.append(test_mse)
    
    return test_MSE

# d
def repeat_MSE(K,iter_num):
    MSE_total = []
    test_MSE_total = []
    for i in range(iter_num):
        MSE = MSE_train(K,seed=i)
        test_MSE = MSE_overfit(K,seed=i)
        MSE_total.append(MSE)
        test_MSE_total.append(test_MSE)
    
    MSE_total = np.array(MSE_total)
    test_MSE_total = np.array(test_MSE_total)

    MSE_avg = np.sum(MSE_total,axis=0)/iter_num
    test_MSE_avg = np.sum(test_MSE_total,axis=0)/iter_num

    return MSE_avg,test_MSE_avg

# plot function for b,c,d
def plot_mse(K,MSE,filename,xname,yname,size=12):
    plt.clf()
    plt.plot(K,np.log(MSE))
    plt.xlabel(xname,fontsize=size)
    plt.ylabel(yname,fontsize=size)
    plt.savefig(filename)

if __name__ == "__main__":
    # a(i)
    sin_grpah()
    
    # a(ii)
    sin_polybase_graph([2,5,10,14,18])

    # b
    K = np.arange(1,19)
    MSE = MSE_train(K, seed = 111)
    plot_mse(K,MSE,"fig2_3.pdf","k","natural log training error")

    # c
    K = np.arange(1,19)
    test_MSE = MSE_overfit(K, seed = 1111)
    plot_mse(K,test_MSE,"fig2_4.pdf","k","natural log testing error")

    # d
    K = np.arange(1,19)
    iter_num = 100
    MSE_avg,test_MSE_avg = repeat_MSE(K,iter_num)
    plot_mse(K,MSE_avg,"fig2_5.pdf","k","average natural log training error",18)
    plot_mse(K,test_MSE_avg,"fig2_6.pdf","k","average natural log testing error",18)
