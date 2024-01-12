import numpy as np
from matplotlib import pyplot as plt

def transform_data_x(data_x,k):
    trans_data = np.ones((len(data_x),k))
    row = 0
    for x in data_x:
        for col in range(k):
            trans_data[row][col] = np.power(x,col)
        row += 1
    return trans_data

def regression_coef(trans_data,data_y):
    temp1 = np.linalg.inv(np.matmul(trans_data.T,trans_data))
    temp2 = np.matmul(trans_data.T,data_y)
    w = np.matmul(temp1,temp2)
    return w

def polynomial_graph(data_x,data_y,k):
    x = np.linspace(0,5,100)
    fig,axes = plt.subplots()
    for i in range(1,k+1):
        trans_data = transform_data_x(data_x,i)
        w = regression_coef(trans_data,data_y)
        print(f"when k={i}, coeffitient:",w)
        mse = sum(np.power(data_y-np.matmul(trans_data,w),2))/len(data_x)
        print(f"when k={i}, mse:",mse)
        # generate polynomial func
        y = 0
        power = 0
        for coef in np.nditer(w):
            y += coef * x ** power
            power += 1
        axes.plot(x,y,linewidth=1.2)
    axes.legend([f"k={i}" for i in range(1,k+1)])
    plt.scatter(data_x,data_y,s=15,c='k')
    plt.savefig("fig1.pdf")

if __name__ == "__main__":
    data_x = np.array([1,2,3,4])
    data_y = np.array([3,2,0,5])
    polynomial_graph(data_x,data_y,k=4)