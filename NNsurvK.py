#Neural network based on a discrete time model (version 3)
# Fused-Lasso Regularization
# Kaplan-Meier censoring indicator
#
#Author: Mathilde Sautreuil, CentraleSupélec, mathilde.sautreuil@gmail.com
#Python version 3.9.2 and Keras version ? (using TensorFlow backend)

# NNsurv
#This is a package for running the neural network based a discrete-time model (version 1).

#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import keras
import keras.backend as K
from keras.engine.topology import Layer
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, LSTM, GRU, Embedding, Concatenate, Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, TimeDistributed
from keras import optimizers, layers, regularizers, initializers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
import math
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from scipy import *
import time
import torch
import torchtuples as tt

from pycox.datasets import metabric
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.evaluation import EvalSurv



import argparse
import fusedRegularizers as fused

parser = argparse.ArgumentParser()
parser.add_argument("fileZ", help="File of ovariate matrix")
parser.add_argument("fileTimes", help="File of time vector")
parser.add_argument("fileCens", help="File of censorship vector")
parser.add_argument("h", help="number of the cross-validation")
args = parser.parse_args()

####################
## Utils functions #
####################
## function to create dil matrix

def create_event_table(data, interval_l):
    kmf = KaplanMeierFitter()
    kmf.fit(durations = data.time, event_observed = data.dead)
    time_ordered = kmf.event_table.index.values
    observed = kmf.event_table.observed.values
    at_risk = kmf.event_table.at_risk.values
    new_time_ordered = [0]
    new_observed = [0]
    new_at_risk = [at_risk[0]]
    keep_index = [0]
    for i in np.arange(1,len(interval_l)):
        keep_index.append(np.int(np.where(time_ordered<interval_l[i])[0][-1:]))
        new_observed.append(np.sum(observed[keep_index[i-1]:keep_index[i]]))
        new_time_ordered.append(interval_l[i])
        new_at_risk.append(np.float(at_risk[np.where(time_ordered<interval_l[i])][-1:]))

    new_observed_arr = np.array(new_observed).reshape((-1,1))
    new_time_ordered_arr = np.array(new_time_ordered).reshape((-1,1))
    new_at_risk_arr = np.array(new_at_risk).reshape((-1,1))
    new_event_table = np.concatenate((new_time_ordered_arr, new_observed_arr, new_at_risk_arr), axis = 1)
    new_event_table_df = pd.DataFrame(new_event_table)
    new_event_table_df.columns = ['time', 'n_observed', 'n_risk']
    cik = np.divide(new_observed, new_at_risk)
    return (cik, new_event_table_df)


def create_dil_init(ystatus, ytime):
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

def create_dil(ystatus, ytime, tau, interval_l, cik):
    n_ind = len(ystatus)
    l = len(interval_l)
    dil = np.zeros((n_ind, l))
    i = 0
    n_all = len(ytime)
    while (i < n_all):
        if (ystatus[i] == 1):
            dil[i, (np.where(ytime[i] >= interval_l)[0][-1])] = 1
            dil[i, (np.where(ytime[i] >= interval_l)[0][-1] + 1):] = 1 #'nan'
        else:
            dil[i, (np.where(ytime[i] >= interval_l)[0][-1] + 1):] = cik[(np.where(ytime[i] >= interval_l)[0][-1] + 1):]
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
    interval_l = tau * (1. / pas) * np.arange(start = 0, stop = pas)
    #al = np.tile(interval_l, x.shape[0])
    g = interval_l
    tl = interval_l[:-1]
    #tl_bis = interval_l
    tl1 = interval_l[1:]
    #tl1_bis = np.concatenate(tl1, interval_l[-1])
    al = (tl + tl1)/2.
    al = np.append(al, interval_l[-1])
    x_long = np.repeat(x, dil.shape[1], axis=0)
    al_long = np.tile(al, n_ind)
    al_long = al_long.reshape([x_long.shape[0],-1])

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

    x_long_al = np.concatenate([x_long, al_long], axis=1)
    x_long_train = x_long_al[trainVF,:]
    x_long_valid = x_long_al[validVF,:]
    x_long_test = x_long_al[testVF,:]

    return (x_long_train, x_long_valid, x_long_test)



def surv_likelihood_loss(y_true, y_pred):
    """Create custom Keras loss function for neural network survival model.
    Required to have only 2 arguments by Keras.
    Arguments
        y_true: Tensor.
          output of make_yik_array
        y_pred: Tensor, predicted hazard for each time interval.
    Returns
        Vector of losses for this minibatch.
    """
    #~ pos_na = np.where(np.isnan(y_true))
    #~ ytrue1 = y_true
    #~ ytrue2 = y_true
    #~ ytrue1[pos_na[0],pos_na[1]] = 0
    #~ ytrue2[pos_na[0],pos_na[1]] = 1
    #~ y_pred.reshape([y_true.shape[0], y_true.shape[1]])
    ind_bool = K.cast(K.equal(y_true,-1),K.floatx())
    ytrue1 = y_true*(1-ind_bool)
    ytrue2 = y_true*(ind_bool)*(-1) # + (1-ind_bool) -> to check this one !!
    #~ ytrue1[ind_bool[0],ind_bool[1]] = 0
    #~ ytrue2[ind_bool[0],ind_bool[1]] = 1

    return K.sum(-y_true * K.log(K.clip(y_pred,K.epsilon(),None)) - (1 - y_true) * K.log(K.clip(1-y_pred,K.epsilon(),None)),axis=None) #-1#return -log likelihood

fileZ = args.fileZ
fileTimes = args.fileTimes
fileCens = args.fileCens
h = np.int(args.h)
#fileZ = "data/simuZ_vf_KIRC.csv"
#fileTimes = "data/surv_time_vf_KIRC.csv"
#fileCens = "data/right_cens_vf_KIRC.csv"
#h = 0
# Creation of the test set and global set
# and Visualization
x_df = pd.read_csv(fileZ)
x = x_df.values
x = x.astype("float64")
#x = x[:,0:10]
print(x.shape[0])


ytime_df = pd.read_csv(fileTimes)
ytime = ytime_df.values
ytime = ytime.astype("float32")

ystatus_df = pd.read_csv(fileCens)
ystatus = ystatus_df.values

dil = create_dil_init(ystatus,ytime)
#np.savetxt('../PLANN/DATA/dil_KIRC_vf.csv', dil, delimiter=',')


x_train, ytime_train, ystatus_train, dil_train, x_valid, ytime_valid, ystatus_valid, dil_valid, x_test, ytime_test, ystatus_test, dil_test = create_split_train_valid(x = x, ytime = ytime, ystatus = ystatus, dil = dil)

dil_train_2d = np.repeat(dil_train, dil_train.shape[1], axis = 0)
dil_valid_2d = np.repeat(dil_valid, dil_valid.shape[1], axis = 0)
dil_test_2d = np.repeat(dil_test, dil_test.shape[1], axis = 0)
dil_train_valid_2d = np.concatenate((dil_train_2d, dil_valid_2d), axis = 0)
dil_train_valid = np.concatenate((dil_train, dil_valid), axis = 0)
pos_na = np.where(np.isnan(dil_train_2d))
dil_train_2d[pos_na]=-1
#print(dil_train_2d)
print(dil_train_2d.shape)
pos_na = np.where(np.isnan(dil_valid_2d))
dil_valid_2d[pos_na]=-1
dil_train_valid_2d = np.concatenate((dil_train_2d, dil_valid_2d), axis = 0)


#x_train_2d, x_valid_2d, x_test_2d = create_x_long(x, ytime, pas=100)
#print(x_train_2d.shape)
#print(x_valid_2d.shape)

## x_long_test_sep
pas = 100
tau = np.max(ytime)  # + 0.0001
interval_l = tau * (1. / pas) * np.arange(start = 0, stop = pas)

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

#x_train_valid_2d = np.concatenate((x_train_2d, x_valid_2d), axis = 0)
ytime_train_valid = np.concatenate((ytime_train, ytime_valid), axis = 0)
ystatus_train_valid = np.concatenate((ystatus_train, ystatus_valid), axis = 0)
x_train_valid = np.concatenate((x_train, x_valid), axis = 0)
data_train_valid = np.concatenate((ytime_train_valid, ystatus_train_valid, x_train_valid), axis = 1)
data_train_valid = pd.DataFrame(data_train_valid)
data_train_valid.columns = ['time', 'dead'] + ["V" + str(_) for _ in k]


cik_train_valid, event_table_train_valid = create_event_table(data_train_valid, interval_l)
cik_train, event_table_train = create_event_table(data_train, interval_l)
cik_valid, event_table_valid = create_event_table(data_valid, interval_l)

dil_train_valid = create_dil(ystatus = ystatus_train_valid, ytime = ytime_train_valid, tau = tau, interval_l = interval_l, cik = cik_train_valid)
dil_train = create_dil(ystatus = ystatus_train, ytime = ytime_train, tau = tau, interval_l = interval_l, cik = cik_train)
dil_valid = create_dil(ystatus = ystatus_valid, ytime = ytime_valid, tau = tau, interval_l = interval_l, cik = cik_valid)

#print(dil_train_valid_2d)
#print(x_train_valid_2d.shape)
#x_inputs_train_valid = x_train_valid_2d.reshape([-1, x_train_valid_2d.shape[2]])
#print(x_train_2d.shape)
#print(x_train_valid_2d.shape)
#x_inputs_train = x_train_2d.reshape([x_train_2d.shape[0]*x_train_2d.shape[1], -1])
#print(x_train_2d.shape)
#print(x_train_2d.shape)
#print(x_train_2d.shape)
#x_inputs_valid = x_valid_2d.reshape([x_valid_2d.shape[0]*x_valid_2d.shape[1], -1])
#print(x_valid_2d.shape)
#print(x_valid_2d[0:5,:-1])
#print(x_valid_2d[0:5,:])
#x_inputs_test = x_test_2d.reshape([x_test_2d.shape[0]*x_test_2d.shape[1], -1])


#~ pos_na_rep = np.where(np.isnan(dil_train_rep))
#~ dil_train_rep[pos_na_rep[0],pos_na_rep[1]]=-1
def create_model(act, neurons, Lambda, d1, d2, reg, LR, opt):
    model = Sequential()
    if reg == 'l2':
        model.add(Dense(neurons, activation=act,input_dim=x_train.shape[1], kernel_regularizer=regularizers.l2(Lambda), kernel_initializer = initializers.glorot_uniform(seed=1)))
    elif reg == 'l1':
        model.add(Dense(neurons, activation=act,input_dim=x_train.shape[1], kernel_regularizer=regularizers.l1(Lambda), kernel_initializer = initializers.glorot_uniform(seed=1)))
    else:
        model.add(Dense(neurons, activation=act, input_dim=x_train.shape[1], kernel_regularizer=regularizers.l1_l2(l1 = Lambda, l2 = Lambda), kernel_initializer = initializers.glorot_uniform(seed=1)))
    if reg == 'l2':
        model.add(Dense(dil_train.shape[1], activation='sigmoid', kernel_regularizer=fused.l2(Lambda), kernel_initializer = initializers.glorot_uniform(seed=1)))
    elif reg == 'l1':
        model.add(Dense(dil_train.shape[1], activation='sigmoid', kernel_regularizer=fused.l1(Lambda), kernel_initializer = initializers.glorot_uniform(seed=1)))
    else:
        model.add(Dense(dil_train.shape[1], activation='sigmoid', kernel_regularizer=fused.l1_l2(l1 = Lambda, l2 = Lambda), kernel_initializer = initializers.glorot_uniform(seed=1)))
    if opt == 'sgd':
        optim = optimizers.SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=surv_likelihood_loss, optimizer=optim)
    return model

hn = np.int(np.ceil(np.sqrt(x_train.shape[1])))
Lambda = np.exp(np.arange(-6,2, 0.5))
BSLambda = [[10, Lambda[0]], [20, Lambda[0]], [5, Lambda[0]], [3, Lambda[0]], [35, Lambda[0]],
            [10, Lambda[1]], [20, Lambda[1]], [5, Lambda[1]], [3, Lambda[1]],[35, Lambda[1]],
            [10, Lambda[2]], [20, Lambda[2]], [5, Lambda[2]], [3, Lambda[2]],[35, Lambda[2]]]

#BSLambda[h][0] = 1
n_train_valid = x_train.shape[0] + x_valid.shape[0]
print(n_train_valid)
perm = np.arange(n_train_valid)
size = np.int(n_train_valid/5)
scores_train = 0
scores_test = 0
dict_cv_model = {}
dict_cv_history = {}
dict_cv_cindex_train = {}
dict_cv_cindex_valid = {}
cta = []
cte = []
ctda = []
ctde = []
for k in np.arange(5):
    key = "k" + str(k)
    rank = perm[(k*size):((k+1)*size)]
    print(rank)
    dataTrain = data_train_valid.drop(rank, axis = 0)
    x_train_NN = np.delete(x_train_valid, rank, 0)
    print(x_train_NN.shape)
    y_train_NN = np.delete(dil_train_valid, rank, 0)
    x_valid_NN = x_train_valid[rank,:]
    y_valid_NN = dil_train_valid[rank,:]
    dataValid = data_train_valid.loc[rank]
    model = create_model('sigmoid', hn, BSLambda[h][1], 0.0, 0.0, 'l2', 0.0001, 'adam')
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights = True, verbose = 2, min_delta = 0.0002)
    #progbar = generic_utils.Progbar(target = None, interval = 0.001)
    history = model.fit(x_train_NN, y_train_NN, batch_size = BSLambda[h][0], epochs=5, validation_data = (x_valid_NN, y_valid_NN), callbacks=[early_stopping], verbose = 1)
    dict_cv_model[key] = model
    dict_cv_history[key] = history.history
    #for layer in model.layers:
    #    g=layer.get_config()
    #    w=layer.get_weights()
    #    print(w)
    ypred_train_NN = model.predict_proba(x_train_NN)
    ypred_test_NN = model.predict_proba(x_valid_NN)
    y_pred_valid_surv = np.cumprod((1-ypred_test_NN), axis = 1)
    y_pred_train_surv = np.cumprod((1-ypred_train_NN), axis = 1)
    oneyr_surv_train = y_pred_train_surv[:,50]
    oneyr_surv_valid = y_pred_valid_surv[:,50]
    surv_valid = pd.DataFrame(np.transpose(y_pred_valid_surv))
    surv_valid.index = interval_l
    surv_train = pd.DataFrame(np.transpose(y_pred_train_surv))
    surv_train.index = interval_l
    dict_cv_cindex_train[key] = concordance_index(dataTrain.time,oneyr_surv_train)
    dict_cv_cindex_valid[key] = concordance_index(dataValid.time,oneyr_surv_valid)
    ev_valid = EvalSurv(surv_valid, dataValid['time'].values, dataValid['dead'].values, censor_surv='km')
    scores_test += ev_valid.concordance_td()
    ev_train = EvalSurv(surv_train, dataTrain['time'].values, dataTrain['dead'].values, censor_surv='km')
    scores_train += ev_train.concordance_td()
    cta.append(concordance_index(dataTrain.time,oneyr_surv_train))
    cte.append(concordance_index(dataValid.time,oneyr_surv_valid))
    ctda.append(ev_train.concordance_td())
    ctde.append(ev_valid.concordance_td())
    #scores_train += concordance_index(dataTrain.time,oneyr_surv_train)
    #scores_test += concordance_index(dataValid.time,oneyr_surv_valid)

save_loss_png = "loss_cv_" + str(h) + "_KIRC_vK.png"
import seaborn as sns
fs = 20
plt.rc('axes', facecolor = "white", linewidth = 1,
       grid = False, edgecolor = "black", titlesize = fs, labelsize = fs)
plt.rc('font', size = fs)
plt.rc('xtick', labelsize = fs)
plt.rc('ytick', labelsize = fs)
plt.rc('legend', fontsize = fs)
plt.rc('figure', titlesize = fs, figsize = (15, 10))

y = BSLambda[h][0]*100 - BSLambda[h][0]*10
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def get_color(i): return colors[np.mod(i, len(colors))]
keys = ["k" + str(_) for _ in np.arange(5)]
for i in range(0,len(keys)):
    plt.plot(dict_cv_history[keys[i]]['loss'],
             color = get_color(i), label = keys[i]+"_train")
    plt.plot(dict_cv_history[keys[i]]['val_loss'],
             linestyle = "--", color = get_color(i), label = keys[i]+"_valid")
plt.xlim(0, 1999)
plt.ylim(0, y)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc = "best", fontsize = 10)
plt.savefig(save_loss_png)


scores_train /= 5
scores_test /= 5
cv_res = np.concatenate((Lambda[h],scores_train,scores_test), axis = None)
cv_res_df = pd.DataFrame(cv_res)
ci_cv_train = []
ctd_cv_train = []
ci_cv_test = []
ctd_cv_test = []
model = create_model('sigmoid', hn, BSLambda[h][1], 0.0, 0.0, 'l2', 0.0001, 'adam')
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights = True, verbose = 2, min_delta = 0.0002)
history=model.fit(x_train, dil_train, batch_size=BSLambda[h][0], epochs=5, validation_data = (x_valid, dil_valid), callbacks=[early_stopping])
y_pred=model.predict_proba(x_train,verbose=0)
#for layer in model.layers:
#    g=layer.get_config()
#    w=layer.get_weights()
#    print (g)
#    print (w)
y_pred_train_surv = np.cumprod((1-y_pred), axis = 1)
n_test = x_test.shape[0]
oneyr_surv_train = y_pred_train_surv[:,50]
ci_cv_train.append(concordance_index(data_train.time,oneyr_surv_train))
surv_train = pd.DataFrame(np.transpose(y_pred_train_surv))
ev_train = EvalSurv(surv_train, data_train['time'].values, data_train['dead'].values, censor_surv='km')
ctd_cv_train.append(ev_train.concordance_td())
for k in np.arange(5):
    perm = np.arange(n_test)
    np.random.shuffle(perm)
    data_test_shuffle = data_test.loc[perm]
    x_test_shuffle = x_test[perm,:]
    y_pred_test=model.predict_proba(x_test_shuffle,verbose=0)
    y_pred_test_surv = np.cumprod((1-y_pred_test), axis = 1)
    oneyr_surv_test = y_pred_test_surv[:,50]
    ci_cv_test.append(concordance_index(data_test_shuffle.time,oneyr_surv_test))
    surv_test = pd.DataFrame(np.transpose(y_pred_test_surv))
    ev_test = EvalSurv(surv_test, data_test['time'].values, data_test['dead'].values, censor_surv='km')
    ctd_cv_test.append(ev_test.concordance_td())


ci_train = np.mean(ci_cv_train)
ci_test = np.mean(ci_cv_test)
#print(ci_train) #0.5
#print("no mean ci_test", ci_cv_test)
#print(ci_test) #0.5
file = open('output/fichier_KIRC_vK.csv','a')
file.write(str(h) + "_10_0.0002_gU," + 'sigmoid' + "," + str(hn) + "," + str(BSLambda[h][0]) + "," + str(BSLambda[h][1]) + ","  + "0.0" + "," + "0.0" + "," + "l2" + "," + "0.0001" + "," + "adam"+","+ str(scores_train) +","+ str(scores_test) + ","+ str(ci_test) + "," + str(ci_cv_test[0]) + "\n")
#'str(h) + "," + str(cv_res[0]) + "," + str(cv_res[1]) + ","+ str(cv_res[2]) + "," + str(cv_res[3]) + "," + str(cv_res[4]) + ","+ str(cv_res[5]) + ","+ str(cv_res[6]) + ","+ str(cv_res[7]) + ","+ str(scores_train) +","+ str(scores_test) + ","+ str(ci_train) + "," + str(ci_test) + "\n")
file.close()

print("------")
print("-------------------------------------------")
print("------Résultats CV hyperparamètres------------")
print("---res Ctd---")
print(scores_train)
print(scores_test)
print("------")
print("---res Ctd---")
print(ctda)
print(ctde)
print("------")
print("---res Cindex---")
print(cta)
print(cte)
print("-------------------------------------------")
print("------Résultats CV réseaux------------")
print("---res Cindex---")
print("Cindex_train : ", ci_cv_train)
#print("Cindex_valid : ", best_cindex_valid)
print("Cindex_test : ", ci_cv_test)
print("---res Ctd---")
print("Ctd_train : ", ctd_cv_train)
#print("Ctd_valid : ", best_ctd_valid)
print("Ctd_test : ", ctd_cv_test)

ci_train = np.mean(ci_cv_train)
ci_test = np.mean(ci_cv_test)

print("-------------------------------------------")
print(cta)
print(cte)
print("-------------------------------------------")
print(ci_cv_train)
print(ci_cv_test)

## to add to NNsurv files on fusion
file_preds_train = "output/pred_files/preds_NNsurv_train_KIRC_vK_" + str(h) + ".csv"
##file_preds_test = "DATA/output/preds_NNsurv_test_CM_" + str(j) + ".csv"
file_preds_test = "output/pred_files/preds_NNsurv_test_KIRC_vK_" + str(h) + ".csv"
df_preds_NNsurv = pd.DataFrame(y_pred_test_surv)
df_preds_NNsurv_train = pd.DataFrame(y_pred_train_surv)
#g0 = np.arange(0,pas)
#header = [ "a" + str(_) for _ in g0]
#print(df_preds_NNsurv.shape)
#print(len(header))
#df_preds_NNsurv.columns = header
export_preds_NNsurv = df_preds_NNsurv
export_preds_NNsurv.to_csv(file_preds_test,index=False)
#df_preds_NNsurv_train.columns = header
export_preds_NNsurv_train = df_preds_NNsurv_train
export_preds_NNsurv_train.to_csv(file_preds_train,index=False)

