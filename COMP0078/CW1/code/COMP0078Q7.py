import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

def random_H(num,seed):
    np.random.seed(seed)
    random_x = np.random.uniform(0,1,(num,2))
    random_y = np.random.randint(0,2,num)
    return random_x,random_y

def prob_distr(num,random_x,random_y,seed):
    np.random.seed(seed)
    x = np.random.uniform(0,1,(num,2))
    y = np.random.randint(0,2,num)
    prob = np.random.binomial(size=num,n=1,p=0.8) # simulate coin toss
    y[np.where(prob==1)] = predict_label(random_x,random_y,x[np.where(prob==1)],3)
    S = x
    S = np.append(S,y.reshape((-1,1)),axis=1)
    return S

def predict_label(train_x,train_y,predictSet,k):
    dist_matrix = cdist(predictSet,train_x)
    sorted_indx = np.argsort(dist_matrix)[:,:k]
    train_y_matrix = np.repeat(train_y.reshape((1,-1)),len(predictSet),axis=0)
    labels = np.take_along_axis(train_y_matrix,sorted_indx,axis=1)
    count1 = np.count_nonzero(labels==1,axis=1)
    count0 = k-count1
    labelSet = np.zeros(len(predictSet))
    labelSet[count1>count0] = 1
    labelSet[count1==count0] = np.random.randint(0,2,1) # randomly generate either 0 or 1
    return labelSet

def generalisation_error(train_x,train_y,test_x,test_y,k):
    pred_label = predict_label(train_x,train_y,test_x,k)
    error = np.mean(pred_label != test_y)
    return error

K = np.arange(1,50,1)
runs = 100

Error = []
for k in K:
    for i in range(runs):
        random_x,random_y = random_H(100,seed=i)
        training = prob_distr(4000,random_x,random_y,seed=i)
        testing = prob_distr(1000,random_x,random_y,seed=i)
        error = generalisation_error(training[:,:2],training[:,2],testing[:,:2],testing[:,2],k)
        Error.append(error)
        print("current k: "+str(k)+", current i:"+str(i))

Error = np.array(Error).reshape((len(K),runs))
avgError = np.sum(Error,axis=1)/runs

plt.plot(K,avgError)
plt.xlabel("K")
plt.ylabel("Generalisation error")
plt.savefig("KNNfig_7.pdf")