import numpy as np
import matplotlib.pyplot as plt
import random


def generate_problem(n, d, s, std=0.06):
    assert s % 2 == 0, "s needs to be divisible by 2"
    xsp = 0.5 * (np.random.rand(s // 2) + 1)
    xsn = - 0.5 * (np.random.rand(s // 2) + 1)
    xsparse = np.hstack([xsp, xsn, np.zeros(d - s)])
    random.shuffle(xsparse)
    # Generate A
    A = np.random.randn(n, d)
    # Generate eps
    y = A @ xsparse + std * np.random.randn(n)
    return xsparse, A, y

def proximity(x,gamma,lamb):
    rho = gamma * lamb
    return np.sign(x) * np.maximum(0, np.abs(x) - rho)

def lasso(x,y,n,A,lamb):
    return 1/(2*n) * (A@x - y.reshape((-1,1))).T @ (A@x - y.reshape(-1,1)) + lamb * np.linalg.norm(x,ord=1)

def PSGA(x,y,n,A,lamb,iter,is_ergodic=False):
    total_gamma = 0
    xbar_unnormal = np.zeros(x.shape)
    Loss = [lasso(x,y,n,A,lamb).item()]
    for k in range(iter):
        i = np.random.choice(n)
        gamma_k = n / (np.linalg.norm(A)**2 * np.sqrt(k+1))
        x = proximity((x - (gamma_k * (A[i,:].reshape((1,-1))@x - y[i]) * A[i,:]).T), gamma_k, lamb)
        total_gamma += gamma_k
        xbar_unnormal += gamma_k * x
        
        if is_ergodic:
            loss = lasso(xbar_unnormal/total_gamma,y,n,A,lamb)
        else:
            loss = lasso(x,y,n,A,lamb)
        Loss.append(loss.item())

        # # check convergence
        # if k>1 and (Loss[-2] - Loss[-1] < 1e-8):
        #     break
    
    if is_ergodic:
        x = xbar_unnormal/total_gamma

    return x,Loss

def RCPGA(x,y,n,d,A,lamb,iter):
    Loss = [lasso(x,y,n,A,lamb).item()]
    for k in range(iter):
        j = np.random.choice(d)
        gamma_j = n / (np.linalg.norm(A[:,j])**2)
        x[j] = proximity((x[j]-gamma_j/n * A[:,j].reshape((1,-1))@(A@x-y.reshape((-1,1)))),gamma_j,lamb)
        loss = lasso(x,y,n,A,lamb)
        Loss.append(loss.item())
    
    return x,Loss


def plot(x,xsparse,algorithm):
    xsparse = xsparse.reshape(-1)
    x = x.reshape(-1)
    iter = len(x)

    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(10)
    fig.set_figheight(5)
    ax.stem(
        np.arange(d)[np.abs(xsparse) > 0],
        xsparse[np.abs(xsparse) > 0],
        label="$x^*$",
    )
    ax.stem(
        np.arange(d)[np.abs(x) > 0.001],
        x[np.abs(x) > 0.001],
        label=r"$x_{\gamma, \lambda}$",
        linefmt="k:",
        markerfmt="k^",
    )
    ax.axhline(0.0, color="red")
    ax.set_xlim([-10 + 0, d + 10])
    ax.set_ylim([-1.1, 1.1])
    ax.set_xlabel("index $i$")
    ax.set_ylabel(f"$x_i$ (feature_threshold=0.001)")
    ax.legend()
    plt.title(f"Sparse Solution vs Actual ({algorithm})")
    plt.savefig(f"{algorithm}",bbox_inches="tight")
    plt.close()

def plot_loss(losses,legend,algorithm):
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(5)
    for i, loss in enumerate(losses):
        plt.plot(loss, label=legend[i])
    if len(legend) > 1:
        plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Loss ({algorithm})")
    plt.savefig(f"{algorithm}-loss", bbox_inches="tight")
    plt.close()

# generate data
n = 1000
d = 500
s = 50
std = 0.06
xsparse,A,y = generate_problem(n,d,s,std)
x0 = np.random.randn(A.shape[1]).reshape((-1,1))

# PSGA algorithm
lamb = 0.3
iteration = 100000
x_psga,Loss_psga = PSGA(x0,y,n,A,lamb,iteration,is_ergodic=False)

plot(x_psga,xsparse,"PSGA")
plot_loss([Loss_psga],["x"],"PSGA")

# # RCPGA algorithm
lamb = 0.05
iteration = 10000
x_rcpga,Loss_rcpga = RCPGA(x0,y,n,d,A,lamb,iteration)

plot(x_rcpga,xsparse,"RCPGA")
plot_loss([Loss_rcpga],["x"],"RCPGA")

# PSGA with sequence of ergodic mean
lamb = 0.3
iteration = 100000
x_psga_ergo,Loss_psga_ergo = PSGA(x0,y,n,A,lamb,iteration,is_ergodic=True)

plot(x_psga_ergo,xsparse,"PSGA ergodic")
plot_loss([Loss_psga,Loss_psga_ergo],["x","x_ergoic"],"PSGAboth")