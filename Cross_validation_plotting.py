import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize

station_db_panda, storm_para_stack, MaxElev_ori, cluster_total, n_pcs = pd.read_pickle("TRAINING_DATA.pkl")

data = np.array(pd.read_csv('models/CNN_predict_' + str(1) + '.csv').iloc[:,2:4])
TCs_events = storm_para_stack.shape[2]
data_points = data.shape[0]
total_data = np.ones((data_points*TCs_events,data.shape[1]))

for i in range(TCs_events):
    if os.path.isfile(os.path.join('models/CNN_predict_' + str(i+1) + '.csv')):
        data = np.array(pd.read_csv('models/CNN_predict_' + str(i+1) + '.csv'))
        PSS_data = data[:,2:4]
        Bathymetry_data = data[:,6]
        total_data[data_points*i:data_points*(i+1),:] = PSS_data + Bathymetry_data[:, np.newaxis]

filtered_data = total_data[(np.all(total_data > 0, axis=1))]
x = filtered_data[:,0]
y = filtered_data[:,1]

font_size = 15

fig= plt.figure(figsize=(14, 8)) 
ax=plt.subplot(1,1,1)
ax.set_aspect(1)

x_min = np.min(x)
x_max = np.max(x)
y_min = np.min(y)
y_max = np.max(y)

xedges, yedges = np.arange(x_min,x_max, 0.05), np.arange(y_min,y_max, 0.05)
hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
xidx = np.clip(np.digitize(x, xedges), 0, hist.shape[0]-1)
yidx = np.clip(np.digitize(y, yedges), 0, hist.shape[1]-1)
c = hist[xidx, yidx]

idx = c.argsort()
x, y, c = x[idx], y[idx], c[idx]

plt.scatter(x, y, c=c, cmap='jet',s=2)
cb = plt.colorbar(shrink=0.6)
cb.set_label('counts in bin', fontsize = font_size)
cb.ax.tick_params(labelsize=font_size)
plt.plot([x_min,np.floor(x_max)+1],[x_min,np.floor(x_max)+1],color='k',linewidth=1.0,zorder=0)

plt.xlim([x_min,np.floor(x_max)+1])
plt.ylim([x_min,np.floor(x_max)+1])
plt.xlabel(r'$\eta_{ADCIRC}\ (m)$', fontsize = font_size)
plt.ylabel(r'$\eta_{C1PKNet}\ (m)$', fontsize = font_size)
plt.xticks(fontsize = font_size);
plt.yticks(fontsize = font_size);
plt.show()