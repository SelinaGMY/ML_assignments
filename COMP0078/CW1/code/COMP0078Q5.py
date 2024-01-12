import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
import COMP0078Q4

def split_data(data,prop,seed):
    np.random.seed(seed)
    training = data.sample(frac=prop)
    testing = data[~data.index.isin(training.index)]
    return training,testing

def Nfold_split(N,data):
    return np.array(np.array_split(data,N),dtype='object')

def init_params():
    power1 = np.arange(-40,-25,1,dtype="float")
    power2 = np.arange(7,13.5,0.5)
    gamma = 2 ** power1
    sigma = 2 ** power2
    return gamma,sigma

def gaussian_kernel(xi,xj,sigma):
    return np.exp(-sum((xi-xj)**2)/(2*(sigma**2)))

def Kmatrix(train_x,sigma):
    l = len(train_x)
    k_matrix = np.ones((l,l))
    for i in range(l):
        k_matrix[i][i] = 1
        for j in range(i+1,l):
            k_matrix[i][j] = gaussian_kernel(train_x[i],train_x[j],sigma)
            k_matrix[j][i] = k_matrix[i][j]
    return k_matrix

def Alpha(k_matrix,gamma,train_y):
    l = len(train_y)
    I = np.ones((l,l))
    temp = np.linalg.pinv(k_matrix + gamma*l*I)
    return np.matmul(temp,train_y)

def calculate_y_hat(x,xtest,sigma,alpha):
	y_hat = np.zeros(len(xtest))
	for i in range(len(xtest)):
		for j in range(len(x)):
			y_hat[i] += alpha[j]*gaussian_kernel(x[j],xtest[i],sigma)
	return y_hat

def ridge_regression_cv(N,training_cv,gamma,sigma):
    MSE_N = []
    for n in range(N):
        test = training_cv[n]
        test_x = np.array(test.iloc[:,:12])
        test_y = np.array(test["MEDV"])
        train = training[~training.index.isin(test.index)] # 'training' is the whole training set before the cv split
        train_x = np.array(train.iloc[:,:12])
        train_y = np.array(train["MEDV"])

        k_matrix = Kmatrix(train_x,sigma)
        alpha = Alpha(k_matrix,gamma,train_y)

        y_hat = calculate_y_hat(train_x,test_x,sigma,alpha)
        mse = sum((test_y-y_hat)**2) / len(test_y)
        MSE_N.append(mse)
    avg_mse = sum(MSE_N)/len(MSE_N) # average cv mse
    return avg_mse

def mse_calculation_train(train_x,train_y,best_gamma,best_sigma):
    k_matrix = Kmatrix(train_x,best_sigma)
    alpha = Alpha(k_matrix,best_gamma,train_y)
    train_y_hat = calculate_y_hat(train_x,train_x,best_sigma,alpha)
    mse = sum((train_y_hat-train_y)**2)/len(train_y)
    return mse

def mse_calculation_test(train_x,train_y,test_x,test_y,best_gamma,best_sigma):
    k_matrix = Kmatrix(train_x,best_sigma)
    alpha = Alpha(k_matrix,best_gamma,train_y)
    test_y_hat = calculate_y_hat(train_x,test_x,best_sigma,alpha)
    mse = sum((test_y_hat-test_y)**2)/len(test_y)
    return mse

if __name__ == "__main__":
    data = pd.read_csv("Boston-filtered.csv")
    
    N = 5 # N-fold cross validation
    Gamma,Sigma = init_params() # parameter vectors

    training,testing = split_data(data,2/3,seed=1)
    training_x = np.array(training.iloc[:,:12])
    training_y = np.array(training["MEDV"])
    testing_x = np.array(testing.iloc[:,:12])
    testing_y = np.array(testing["MEDV"])
    
    # # a
    record = []
    training_cv = Nfold_split(N,training)
    i = 0 # counter
    for gamma in Gamma:
        for sigma in Sigma:
            avg_mse = ridge_regression_cv(N,training_cv,gamma,sigma)
            record.append((avg_mse,gamma,sigma))
            i+=1
            print(i)

    record = np.array(record)
    best_mse = record[:,0][np.argmin(record[:,0])]
    best_gamma = record[:,1][np.argmin(record[:,0])]
    best_sigma = record[:,2][np.argmin(record[:,0])]

    print("Best gamma: 2^("+str(np.log2(best_gamma))+")")
    print("Best sigma: 2^("+str(np.log2(best_sigma))+")")
    print("Best mse: "+str(best_mse))

    # # b
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'$\sigma$', fontsize=15)
    ax.set_ylabel(r'$\gamma$', fontsize=15)
    ax.set_zlabel('$MSE$', fontsize=15)
    ax.plot_trisurf(record[:,2], record[:,1], record[:,0], cmap=plt.get_cmap('rainbow'))
    plt.savefig("mse_plot.pdf")

    # # c
    training_error = mse_calculation_train(training_x,training_y,best_gamma,best_sigma)
    testing_error = mse_calculation_test(training_x,training_y,testing_x,testing_y,best_gamma,best_sigma)
    print("training error:"+str(training_error))
    print("testing error:"+str(testing_error))

    # d
    # for Q4
    run = 20
    MSE_4a = []
    MSE_4c = []
    MSE_4d = []
    for num in range(run):
        training,testing = split_data(data,2/3,seed=num)
        MSE_4a.append((COMP0078Q4.baseline_regresion(training,testing)))
        MSE_4c_temp = []
        for indx in range(12):
            MSE_4c_temp.append((COMP0078Q4.single_attr_regression(training,testing,indx)))
        MSE_4c.append(MSE_4c_temp)
        MSE_4d.append((COMP0078Q4.all_attr_regression(training,testing)))
    
    mean_4a = np.round(np.sum(np.array(MSE_4a),axis=0)/run,2)
    var_4a = np.round(np.var(np.array(MSE_4a),axis=0),2)
    print("mean for 4a train: "+str(mean_4a[0]))
    print("mean for 4a test: "+str(mean_4a[1]))
    print("var for 4a train: "+str(var_4a[0]))
    print("var for 4a test: "+str(var_4a[1]))

    mean_4c = np.round(np.sum(np.array(MSE_4c),axis=0)/run,2)
    var_4c = np.round(np.var(np.array(MSE_4c),axis=0),2)
    print("mean for 4c train: "+str(mean_4c[:,0]))
    print("mean for 4c test: "+str(mean_4c[:,1]))
    print("var for 4c train: "+str(var_4c[:,0]))
    print("var for 4c test: "+str(var_4c[:,1]))

    mean_4d = np.round(np.sum(np.array(MSE_4d),axis=0)/run,2)
    var_4d = np.round(np.var(np.array(MSE_4d),axis=0),2)
    print("mean for 4d train: "+str(mean_4d[0]))
    print("mean for 4d test: "+str(mean_4d[1]))
    print("var for 4d train: "+str(var_4d[0]))
    print("var for 4d test: "+str(var_4d[1]))

    # for Q5
    MSE_5c = []
    params_record = []
    for num in range(run):
        # find best parameters
        training,testing = split_data(data,2/3,seed=num)
        training_x = np.array(training.iloc[:,:12])
        training_y = np.array(training["MEDV"])
        testing_x = np.array(testing.iloc[:,:12])
        testing_y = np.array(testing["MEDV"])
        training_cv = Nfold_split(N,training)
        
        record = []
        i = 0 # counter
        for gamma in Gamma:
            for sigma in Sigma:
                avg_mse = ridge_regression_cv(N,training_cv,gamma,sigma)
                record.append((avg_mse,gamma,sigma))
                i+=1
                print("current i: "+str(i)+", current run: "+str(num))
        record = np.array(record)
        best_gamma = record[:,1][np.argmin(record[:,0])]
        best_sigma = record[:,2][np.argmin(record[:,0])]
        params_record.append((best_gamma,best_sigma))

        # calculate mses
        training_error = mse_calculation_train(training_x,training_y,best_gamma,best_sigma)
        testing_error = mse_calculation_test(training_x,training_y,testing_x,testing_y,best_gamma,best_sigma)
        MSE_5c.append((training_error,testing_error))
    
    print("20 best gamma and sigma pairs: "+str(params_record))
    mean_5c = np.around(np.sum(np.array(MSE_5c),axis=0)/run,2)
    var_5c = np.round(np.var(np.array(MSE_5c),axis=0),2)
    print("mean for 5c train: "+str(mean_5c[0]))
    print("mean for 5c test: "+str(mean_5c[1]))
    print("var for 5c train: "+str(var_5c[0]))
    print("var for 5c test: "+str(var_5c[1]))