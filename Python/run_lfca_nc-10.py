#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 17:58:52 2018

@author: Zhaoyi.Shen
"""

import sys
# sys.path.append('/home/z1s/py/lib/')
from signal_processing import lfca
import numpy as np
import scipy as sp
from scipy import io
from matplotlib import pyplot as plt

from netCDF4 import Dataset,num2date
from datetime import datetime, timedelta
import csv 
with Dataset('/export/data1/rccheng/ERSSTv5/sst.mnmean.nc') as f:
    lat_axis = f['lat'][:]
    lon_axis = f['lon'][:]
    er_time = f['time'][:]
    er_sst = f['sst'][:]
er_sst[er_sst<-9e36] = np.nan
er_refstart = [(datetime(1800,1,1) + timedelta(days= i)-datetime(1900,1,1)).days for i in er_time]
er_refend = [(datetime(1800,1,1) + timedelta(days= i)-datetime(2020,7,1)).days for i in er_time]
sst = er_sst[er_refstart.index(0):er_refend.index(0)+1,:,:].transpose([2,1,0])
time = np.arange(1900,2020.99,1/12.)[:-5]
nlon = sst.shape[0]
nlat = sst.shape[1]
ntime = sst.shape[2]

# filename = '/export/data1/rccheng/ERSSTv5/ERSST_1900_2016.mat'
# mat = io.loadmat(filename)

# lat_axis = mat['LAT_AXIS']
# lon_axis = mat['LON_AXIS']
# sst = mat['SST']
# nlon = sst.shape[0]
# nlat = sst.shape[1]
# ntime = sst.shape[2]
# time = np.arange(1900,2016.99,1/12.)

cutoff = 120
truncation = 10
#%%
mean_seasonal_cycle = np.zeros((nlon,nlat,12))
sst_anomalies = np.zeros((nlon,nlat,ntime))
for i in range(12):
    mean_seasonal_cycle[...,i] = np.nanmean(sst[...,i:ntime:12],-1)
    sst_anomalies[...,i:ntime:12] = sst[...,i:ntime:12] - mean_seasonal_cycle[...,i][...,np.newaxis]
#%%
s = sst_anomalies.shape
y, x = np.meshgrid(lat_axis,lon_axis)
area = np.cos(y*np.pi/180.)
area[np.where(np.isnan(np.mean(sst_anomalies,-1)))] = 0
#%%
domain = np.ones(area.shape)
domain[np.where(x<100)] = 0
domain[np.where((x<103) & (y<5))] = 0
domain[np.where((x<105) & (y<2))] = 0
domain[np.where((x<111) & (y<-6))] = 0
domain[np.where((x<114) & (y<-7))] = 0
domain[np.where((x<127) & (y<-8))] = 0
domain[np.where((x<147) & (y<-18))] = 0      
domain[np.where(y>70)] = 0
domain[np.where((y>65) & ((x<175) | (x>200)))] = 0
domain[np.where(y<-45)] = 0
domain[np.where((x>260) & (y>17))] = 0
domain[np.where((x>270) & (y<=17) & (y>14))] = 0
domain[np.where((x>276) & (y<=14) & (y>9))] = 0
domain[np.where((x>290) & (y<=9))] = 0
#%%
order = 'C'
x = np.transpose(np.reshape(sst_anomalies,(s[0]*s[1],s[2]),order=order))
area_weights = np.transpose(np.reshape(area,(s[0]*s[1],1),order=order))
domain = np.transpose(np.reshape(domain,(s[0]*s[1],1),order=order))
icol_ret = np.where((area_weights!=0) & (domain!=0))
icol_disc = np.where((area_weights==0) | (domain==0))
x = x[:,icol_ret[1]]
area_weights = area_weights[:,icol_ret[1]]
normvec = np.transpose(area_weights)/np.sum(area_weights)
scale = np.sqrt(normvec)
#%%
lfcs, lfps, weights, r, pvar, pcs, eofs, ntr, pvar_slow, pvar_lfc, r_eofs, pvar_slow_eofs = lfca(x, cutoff, truncation, scale)
#%%
nins = np.size(icol_disc[1])
nrows = lfps.shape[0]
lfps_aug = np.zeros((nrows,lfps.shape[1]+nins))
lfps_aug[:] = np.nan
lfps_aug[:,icol_ret[1]] = lfps
nrows = eofs.shape[0]
eofs_aug = np.zeros((nrows,eofs.shape[1]+nins))
eofs_aug[:] = np.nan
eofs_aug[:,icol_ret[1]] = eofs
#%%
s1 = np.size(lon_axis)
s2 = np.size(lat_axis)
i = 1
pattern = np.reshape(lfps_aug[i,...],(s1,s2),order=order)
pattern[np.where(np.abs(pattern)>1.e5)] = np.nan
plt.figure()
plt.contourf(np.squeeze(lon_axis),np.squeeze(lat_axis),np.transpose(pattern),\
             np.arange(-1,1.1,0.1),cmap=plt.cm.RdYlBu_r)
plt.savefig('./nc-lfp-10-1900-2016.png')
plt.figure()
plt.bar(time,lfcs[:,i])
plt.savefig('./nc-lfc-10-1900-2016.png')

output = np.concatenate([lfcs[:,i],np.array([np.nan,np.nan,np.nan,np.nan,np.nan])])
with open('/home/rccheng/Documents/stormtrack/indices/LFC-PDO-10-1900-2020.txt','w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(output)
    