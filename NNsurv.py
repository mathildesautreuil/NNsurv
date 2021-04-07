#!/usr/bin/env python
# coding: utf-8

# # Creation of data files for NNsurv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument("h", help="number of the cross-validation")
#args = parser.parse_args()

#np.random.seed(1337)
#h = np.int(args.h)

####################
## Utils functions #
####################
## function to create dil matrix
def create_dil(ystatus, ytime):
    n_ind = len(ystatus)
    tau = np.max(ytime)  # + 0.0001
    pas = 100
    interval_l = tau * (1 / pas) * np.arange(start = 0, stop = pas)
    l = len(interval_l) ## nombre de discretitation temps
    dil = np.zeros((n_ind, l))
    i = 0
    n_all = len(ytime)
    while (i < n_all):
        if (ystatus[i] == 1):
            dil[i, (np.where(ytime[i] >= interval_l)[0][-1])] = 1
            dil[i, (np.where(ytime[i] >= interval_l)[0][-1] + 1):] = 'nan'
        else:
            dil[i, (np.where(ytime[i] >= interval_l)[0][-1] + 1):] = 'nan'
        i = i + 1
                   
    return dil

def create_split_train_valid(x, ytime, ystatus, dil):
    n_ind = x.shape[0]
    ind = np.arange(x.shape[0])

    n_train = int(70*n_ind/100)
    n_valid = int(10*n_ind/100)
    n_test =  int(20*n_ind/100)
    train = ind[0:n_train]
    valid = ind[n_train:(n_train+n_valid)]
    test = ind[(n_train+n_valid):(n_train+n_valid+n_test)]

    
    x_train = x[train,:]
    ytime_train = ytime[train,:]
    ystatus_train = ystatus[train,:]
    dil_train = dil[train,:]
    
    x_valid = x[valid,:]
    ytime_valid = ytime[valid,:]
    ystatus_valid = ystatus[valid,:]
    dil_valid = dil[valid,:]

    x_test = x[test,:]
    ytime_test = ytime[test,:]
    ystatus_test = ystatus[test,:]
    dil_test = dil[test,:]

    
    return (x_train, ytime_train, ystatus_train, dil_train, x_valid, ytime_valid, ystatus_valid, dil_valid, x_test, ytime_test, ystatus_test, dil_test)
    
## x: give the dataset without the test (n=1000 normally or 200 or 500)    
def create_x_long(x, ytime, pas):
    n_ind = x.shape[0]
    ind = np.arange(x.shape[0])
    tau = np.max(ytime)  # + 0.0001
    interval_l = tau * (1 / pas) * np.arange(start = 0, stop = pas)
    al = np.tile(interval_l, x.shape[0])
    al = al.reshape([al.shape[0],1])
    x_long = np.repeat(x, dil.shape[1], axis=0)
   
    n_train = int(70*n_ind/100)
    n_valid = int(10*n_ind/100)
    n_test =  int(20*n_ind/100)
    train = ind[0:n_train]
    trainVF = []
    for i in ind:
      trainVF.append(np.arange((i)*pas, (i+1)*pas))
    valid = ind[n_train:(n_train+n_valid)]
    validVF = []
    for i in ind:
      validVF.append((np.arange((i)*pas, (i+1)*pas)))
    test  = ind[(n_train + n_valid):(n_train + n_valid + n_test)]
    testVF = []
    for i in ind:
      testVF.append((np.arange((i)*pas, (i+1)*pas)))
    trainVF= trainVF[0:n_train]
    validVF = validVF[n_train:(n_train+n_valid)]
    testVF = testVF[(n_train+n_valid):(n_train+n_valid+n_test)]
    
    x_long_al = np.concatenate([x_long, al], axis=1)
    x_long_train = x_long_al[trainVF,:]
    x_long_valid = x_long_al[validVF,:]
    x_long_test = x_long_al[testVF,:]

    return (x_long_train, x_long_valid, x_long_test)
 


fileZ = "../PLANN/DataFromR/simuZ_vf_KIRC.csv"
fileTimes = "../PLANN/DataFromR/surv_time_vf_KIRC.csv"
fileCens = "../PLANN/DataFromR/right_cens_vf_KIRC.csv"
# Creation of the test set and global set
# and Visualization
x_df = pd.read_csv(fileZ)
x = x_df.values
x = x.astype("float64")
print(x.shape[0])


ytime_df = pd.read_csv(fileTimes)
ytime = ytime_df.values
ytime = ytime.astype("float32")

ystatus_df = pd.read_csv(fileCens)
ystatus = ystatus_df.values
# Creation of the test set and global set
# and Visualization 
#np.savetxt("../PLANN/DATA/ytime_test_KIRC_vf.csv", ytime_test, delimiter=',')
#np.savetxt('../PLANN/DATA/ystatus_test_KIRC_vf.csv', ystatus_test, delimiter=',')

dil = create_dil(ystatus,ytime)
#np.savetxt('../PLANN/DATA/dil_KIRC_vf.csv', dil, delimiter=',')


x_train, ytime_train, ystatus_train, dil_train, x_valid, ytime_valid, ystatus_valid, dil_valid, x_test, ytime_test, ystatus_test, dil_test = create_split_train_valid(x = x, ytime = ytime, ystatus = ystatus, dil = dil)

dil_train_2d = np.repeat(dil_train, dil_train.shape[1], axis = 0)
dil_valid_2d = np.repeat(dil_valid, dil_valid.shape[1], axis = 0)
dil_test_2d = np.repeat(dil_test, dil_test.shape[1], axis = 0)
dil_train_valid_2d = np.concatenate((dil_train_2d, dil_valid_2d), axis = 0)
dil_train_valid = np.concatenate((dil_train, dil_valid), axis = 0)


x_train_2d, x_valid_2d, x_test_2d = create_x_long(x, ytime, pas=100)
#print(x_train_2d.shape)
#print(x_valid_2d.shape)

## x_long_test_sep
pas = 100

# Creation of dataframes
data_train = np.concatenate((ytime_train, ystatus_train, x_train), axis = 1)
data_test = np.concatenate((ytime_test, ystatus_test, x_test), axis = 1)
data_valid = np.concatenate((ytime_valid, ystatus_valid, x_valid), axis = 1)
data_train = pd.DataFrame(data_train)
data_test = pd.DataFrame(data_test)
data_valid = pd.DataFrame(data_valid)
k = np.arange(x_train.shape[1])
data_train.columns = ['time', 'dead'] + ["V" + str(_) for _ in k]
data_valid.columns = ['time', 'dead'] + ["V" + str(_) for _ in k] #['time', 'dead', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']
data_test.columns = ['time', 'dead'] + ["V" + str(_) for _ in k]

x_train_valid_2d = np.concatenate((x_train_2d, x_valid_2d), axis = 0)
ytime_train_valid = np.concatenate((ytime_train, ytime_valid), axis = 0)
ystatus_train_valid = np.concatenate((ystatus_train, ystatus_valid), axis = 0)
x_train_valid = np.concatenate((x_train, x_valid), axis = 0)
data_train_valid = np.concatenate((ytime_train_valid, ystatus_train_valid, x_train_valid), axis = 1)
data_train_valid = pd.DataFrame(data_train_valid)
data_train_valid.columns = ['time', 'dead'] + ["V" + str(_) for _ in k]

#print(dil_train_valid_2d)
#print(x_train_valid_2d.shape)
x_train_valid_2d = x_train_valid_2d.reshape([-1, x_train_valid_2d.shape[2]])
#print(x_train_2d.shape)
#print(x_train_valid_2d.shape)
x_train_2d = x_train_2d.reshape([x_train_2d.shape[0]*x_train_2d.shape[1], -1])
#print(x_train_2d.shape)
#print(x_train_2d.shape)
#print(x_train_2d.shape)
x_valid_2d = x_valid_2d.reshape([x_valid_2d.shape[0]*x_valid_2d.shape[1], -1])
#print(x_valid_2d.shape)
np.savetxt('../PLANN/DATA/x_train_2d_KIRC_vf.csv', x_train_2d, delimiter=",")
np.savetxt('../PLANN/DATA/x_valid_2d_KIRC_vf.csv', x_valid_2d, delimiter=",")
#print(x_valid_2d[0:5,:-1])
#print(x_valid_2d[0:5,:])
x_test_2d = x_test_2d.reshape([x_test_2d.shape[0]*x_test_2d.shape[1], -1])
#print(x_test_2d.shape)
np.savetxt('../PLANN/DATA/x_test_2d_KIRC_vf.csv', x_test_2d, delimiter=",")

