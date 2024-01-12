import os
os.chdir(os.path.expanduser('~/ML')) 
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.colors
from scipy.spatial.distance import cdist, pdist
from scipy.stats import mode
# Q6
# random process H and plot
x = np.random.random_sample([100,2])
y = np.random.randint(0,high = 2,size=[100,1]) 
S = np.append(x,y,axis = 1)
x1p,x2p = np.meshgrid(np.linspace(0, 1, len(y)) ,np.linspace(0, 1, len(y)))
x1_flat = x1p.ravel()
x2_flat = x2p.ravel()
x_flat = np.vstack((x1_flat,x2_flat)).T 
xdist = cdist(x_flat, x, metric='euclidean')
v = 3
nearest_xindx = np.argsort(xdist, axis=1)[:, :v]
nearest_y = y[nearest_xindx]
h = mode(nearest_y, axis=1).mode
map_color = {}
map_color[tuple(np.array([1]))] = 'blue'
map_color[tuple(np.array([0]))] = 'olivedrab'
color = [map_color[tuple(item)] for item in y]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['white','darkcyan'])
plt.contourf(x1p,x2p, np.reshape(h,x1p.shape),cmap = cmap)
plt.scatter(x[:,0],x[:,1],c=color,s=20)
plt.xlim((0,1))
plt.ylim((0,1))
# plt.show()
plt.title("Sample data")
plt.savefig("KNNQ6.pdf")