import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def split_data(data,prop,seed):
    np.random.seed(seed)
    train = data.sample(frac=prop)
    test = data[~data.index.isin(train.index)]
    return train,test

def polykernel(xi,xj,d):
    return (xi @ xj.T) ** d

def gaussian_kernel(xi,xj,c):
    diff = np.sum(xi**2, axis=1).reshape(-1, 1) + np.sum(xj**2, axis=1) - 2 * np.dot(xi, xj.T)
    return np.exp(-c * diff)

def train(train_x,train_y,kernel,max_epoch,k,d):
    # initialise
    l = len(train_x)
    Kmatrix = kernel(train_x,train_x,d)
    alpha = np.zeros((k,l))
    train_error = []

    for epoch in range(max_epoch):
        error = 0
        for i,y in enumerate(train_y):
            # prediction
            conf = alpha @ Kmatrix[:,i]
            yhat = np.argmax(conf)
            if yhat != y:
                error += 1
            # update
            for j in range(len(conf)):
                if conf[j] >= 0 and j != y:
                    alpha[j,i] -= 1
                elif conf[j] < 0 and j == y:
                    alpha[j,i] += 1
        train_error.append(error)

        if epoch > 1 and (train_error[-2] - train_error[-1])/l < 1e-6:
            break
    
    return train_error,alpha

def test(train_x,test_x,test_y,alpha,kernel,d):
    l = len(train_x)
    Kmatrix = kernel(train_x,test_x,d)
    yhat = np.argmax(alpha @ Kmatrix, axis=0)
    error = np.sum(yhat != test_y)/l
    
    return error

def confusion_matrix(train_x,test_x,test_y,alpha,kernel,k,d):
    l = len(train_x)
    confusion_count = np.zeros((k,k))
    Kmatrix = kernel(train_x,test_x,d)
    yhats = np.argmax(alpha @ Kmatrix, axis=0)
    for yhat,y in zip(yhats,test_y):
        confusion_count[round(y),round(yhat)] += 1

    confusion_matrix = confusion_count / np.sum(confusion_count,axis=1)
    confusion_matrix[np.diag_indices_from(confusion_matrix)] = 0

    return confusion_matrix

        
# Q1
D = [1,2,3,4,5,6,7]
iter = 20
max_epoch = 20
k = 10

data = pd.DataFrame(np.loadtxt("zipcombo.dat"))

for d in D:
    train_errors = []
    test_errors = []
    for run in range(iter):
        # randomly split data
        training,testing = split_data(data,0.8,seed=run)
        train_y = np.array(training.iloc[:,0])
        train_x = np.array(training.iloc[:,1:])
        test_y = np.array(testing.iloc[:,0])
        test_x = np.array(testing.iloc[:,1:])

        train_error,alpha = train(train_x,train_y,polykernel,max_epoch,k,d)
        train_errors.append(train_error[-1]/len(train_x))
        test_error = test(train_x,test_x,test_y,alpha,polykernel,d)
        test_errors.append(test_error)
    
    print(f"d = {d}, mean train error: {np.mean(train_errors)}, train error std: {np.std(train_errors)}")
    print(f"d = {d}, mean test error: {np.mean(test_errors)}, test error std: {np.std(test_errors)}")

# Q2 & Q3 & Q4
K = 5
X = np.array(data.iloc[:,1:])
Y = np.array(data.iloc[:,0])

optimal_D = []
test_errors = []
cmatrices = []
allerror_count = np.zeros(Y.shape)
for run in range(iter):
    training,testing = split_data(data,0.8,seed=run)
    train_x = np.array(training.iloc[:,1:])
    train_y = np.array(training.iloc[:,0])
    test_x = np.array(testing.iloc[:,1:])
    test_y = np.array(testing.iloc[:,0])

    # Q2: cross validation, find optimal d
    kf_train = np.array_split(training,K)
    avg_val_errors = []
    for d in D:
        val_errors = []
        for kf in range(K):
            testset = kf_train[kf]
            trainset = training[~training.index.isin(testset.index)]
            testset_y = np.array(testset.iloc[:,0])
            testset_x = np.array(testset.iloc[:,1:])
            trainset_y = np.array(trainset.iloc[:,0])
            trainset_x = np.array(trainset.iloc[:,1:])

            trainset_error, kf_alpha = train(trainset_x,trainset_y,polykernel,max_epoch,k,d)
            val_error = test(trainset_x,testset_x,testset_y,kf_alpha,polykernel,d)
            val_errors.append(val_error)
            # display progress
            print(f"current run: {run}, d: {d}, fold: {kf}")
        avg_val_errors.append(np.mean(val_errors))
    
    optimal_d = D[np.argmin(avg_val_errors)]
    optimal_D.append(optimal_d)

    # Q2: compute test error by optimnal d
    train_error,alpha = train(train_x,train_y,polykernel,max_epoch,k,optimal_d)
    test_error = test(train_x,test_x,test_y,alpha,polykernel,optimal_d)
    test_errors.append(test_error)
    
    # Q3: confusion matrix
    cmatrix = confusion_matrix(train_x,test_x,test_y,alpha,polykernel,k,optimal_d)
    cmatrices.append(cmatrix)

    # Q4: hardest to predict
    # predict on the whole set
    _,allalpha = train(X,Y,polykernel,max_epoch,k,optimal_d)
    allKmatrix = polykernel(X,X,optimal_d)
    Yhat = np.argmax(allalpha @ allKmatrix, axis=0)
    allerror_count[Yhat!=Y] += 1


# Q2
print(f"mean test error: {np.mean(test_errors)}, std: {np.std(test_errors)}")
print(f"mean optimal d: {np.mean(optimal_D)}, std: {np.std(optimal_D)}")

# Q3
print(optimal_D)
print(cmatrices)
print("mean confusion matrix:")
print(np.mean(cmatrices,axis=0))
print("corresponding std:")
print(np.std(cmatrices,axis=0))

# Q4 
index = np.argsort(-allerror_count)[:5]
plt.figure(figsize=(8, 8))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(np.reshape(X[index[i]], (16,16)), 
               interpolation="Nearest",
               cmap='gray')
    plt.title(f"true value: {int(Y[index[i]])}", fontsize=10)
    plt.axis('off')
plt.show()

# Q5
# repeat Q1
iter = 20
k = 10
max_epoch = 20
S = math.e ** np.arange(1,-4,-1,dtype='float')

for c in S:
    train_errors = []
    test_errors = []
    for run in range(iter):
        training,testing = split_data(data,0.8,seed=run)
        train_x = np.array(training.iloc[:,1:])
        train_y = np.array(training.iloc[:,0])
        test_x = np.array(testing.iloc[:,1:])
        test_y = np.array(testing.iloc[:,0])

        train_error,alpha = train(train_x,train_y,gaussian_kernel,max_epoch,k,c)
        train_errors.append(train_error[-1]/len(train_x))
        test_error = test(train_x,test_x,test_y,alpha,gaussian_kernel,c)
        test_errors.append(test_error)
    
    print(f"for c: {c}, mean train error: {np.mean(train_errors)}, std: {np.std(train_errors)}")
    print(f"for c: {c}, mean test error: {np.mean(test_errors)}, std: {np.std(test_errors)}")

# repeat Q2
K = 5

optimal_C = []
test_errors = []
for run in range(iter):
    training,testing = split_data(data,0.8,seed=run)
    train_x = np.array(training.iloc[:,1:])
    train_y = np.array(training.iloc[:,0])
    test_x = np.array(testing.iloc[:,1:])
    test_y = np.array(testing.iloc[:,0])

    # Q2: cross validation, find optimal d
    kf_train = np.array_split(training,K)
    avg_val_errors = []
    for c in S:
        val_errors = []
        for kf in range(K):
            testset = kf_train[kf]
            trainset = training[~training.index.isin(testset.index)]
            testset_y = np.array(testset.iloc[:,0])
            testset_x = np.array(testset.iloc[:,1:])
            trainset_y = np.array(trainset.iloc[:,0])
            trainset_x = np.array(trainset.iloc[:,1:])

            trainset_error, kf_alpha = train(trainset_x,trainset_y,gaussian_kernel,max_epoch,k,c)
            val_error = test(trainset_x,testset_x,testset_y,kf_alpha,gaussian_kernel,c)
            val_errors.append(val_error)
            # display progress
            print(f"current run: {run}, c: {c}, fold: {kf}")
        avg_val_errors.append(np.mean(val_errors))
    
    optimal_c = S[np.argmin(avg_val_errors)]
    optimal_C.append(optimal_c)

    # Q2: compute test error by optimnal d
    train_error,alpha = train(train_x,train_y,gaussian_kernel,max_epoch,k,optimal_c)
    test_error = test(train_x,test_x,test_y,alpha,gaussian_kernel,optimal_c)
    test_errors.append(test_error)

print(f"mean test error: {np.mean(test_errors)}, std: {np.std(test_errors)}")
print(f"mean optimal c: {np.mean(optimal_C)}, std: {np.std(optimal_C)}")
