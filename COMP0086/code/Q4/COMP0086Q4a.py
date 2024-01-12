import ssm_kalman
from scipy.linalg import cholesky
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

logdet = lambda A: 2 * np.sum(np.log(np.diag(cholesky(A))))

k = 4
d = 5
X = np.array(pd.read_csv('ssm_spins.txt',sep='  ',header=None,engine='python'))
A = np.array([[np.cos(2*np.pi/180),-np.sin(2*np.pi/180),0,0],
              [np.sin(2*np.pi/180),np.cos(2*np.pi/180),0,0],
              [0,0,np.cos(2*np.pi/90),-np.sin(2*np.pi/90)],
              [0,0,np.sin(2*np.pi/90),np.cos(2*np.pi/90)]]) * 0.99
C = np.array([[1,0,1,0],
              [0,1,0,1],
              [1,0,0,1],
              [0,0,1,1],
              [0.5,0.5,0.5,0.5]])
Q = np.identity(k) - np.matmul(A,A.T)
np.random.seed(1)
y_init = np.random.normal(0,1,k)
Q_init = np.ones((k,k))
R = np.identity(d)

y_hat1, V_hat1, V_joint1, likelihood1 = ssm_kalman.run_ssm_kalman(X.T,y_init,Q_init,A,Q,C,R,mode='filt')
epsilon = 1e-6  # Small positive constant
V_hat1_regularized = [matrix + epsilon * np.eye(matrix.shape[0]) for matrix in V_hat1]
plt.figure(figsize=(5,3))
plt.rcParams['font.size'] = 15
plt.plot(y_hat1.T)
plt.savefig('Q4_1.pdf')
plt.clf()
plt.plot(list(map(logdet,V_hat1_regularized)))
plt.savefig("Q4_2.pdf")

y_hat2, V_hat2, V_joint2, likelihood2 = ssm_kalman.run_ssm_kalman(X.T,y_init,Q_init,A,Q,C,R,mode='smooth')
V_hat2_regularized = [matrix + epsilon * np.eye(matrix.shape[0]) for matrix in V_hat2]
plt.figure(figsize=(5,3))
plt.plot(y_hat2.T)
plt.savefig('Q4_3.pdf')
plt.clf()
plt.plot(list(map(logdet,V_hat2_regularized)))
plt.savefig("Q4_4.pdf")