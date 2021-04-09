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
    y_true: Tensor, output of make_yik_array
    y_pred: Tensor, predicted hazard for each time interval.
    Returns
    Vector of losses for this minibatch.
    """

    ind_bool = K.cast(K.equal(y_true, -1), K.floatx())
    ytrue1 = y_true*(1-ind_bool)
    ytrue2 = y_true*(ind_bool)*(-1)
    ytrue3 = ytrue1 + ytrue2

    return K.sum(-ytrue1 * K.log(K.clip(y_pred,K.epsilon(),None)) - (1 - ytrue3) * K.log(K.clip(1-y_pred,K.epsilon(),None)),axis=None) #-1#return -log likelihood

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

dil = create_dil(ystatus,ytime)
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


x_train_2d, x_valid_2d, x_test_2d = create_x_long(x, ytime, pas=100)
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

x_train_valid_2d = np.concatenate((x_train_2d, x_valid_2d), axis = 0)
ytime_train_valid = np.concatenate((ytime_train, ytime_valid), axis = 0)
ystatus_train_valid = np.concatenate((ystatus_train, ystatus_valid), axis = 0)
x_train_valid = np.concatenate((x_train, x_valid), axis = 0)
data_train_valid = np.concatenate((ytime_train_valid, ystatus_train_valid, x_train_valid), axis = 1)
data_train_valid = pd.DataFrame(data_train_valid)
data_train_valid.columns = ['time', 'dead'] + ["V" + str(_) for _ in k]

#print(dil_train_valid_2d)
#print(x_train_valid_2d.shape)
x_inputs_train_valid = x_train_valid_2d.reshape([-1, x_train_valid_2d.shape[2]])
#print(x_train_2d.shape)
#print(x_train_valid_2d.shape)
x_inputs_train = x_train_2d.reshape([x_train_2d.shape[0]*x_train_2d.shape[1], -1])
#print(x_train_2d.shape)
#print(x_train_2d.shape)
#print(x_train_2d.shape)
x_inputs_valid = x_valid_2d.reshape([x_valid_2d.shape[0]*x_valid_2d.shape[1], -1])
#print(x_valid_2d.shape)
#print(x_valid_2d[0:5,:-1])
#print(x_valid_2d[0:5,:])
x_inputs_test = x_test_2d.reshape([x_test_2d.shape[0]*x_test_2d.shape[1], -1])

def create_model(act, neurons, Lambda, d1, d2, reg, LR, opt, seed = 1):
    model = Sequential()
    #Hidden layers would go here. For this example, using simple linear model with no hidden layers.
    if reg == 'l2':
        model.add(Dense(neurons, activation=act,input_dim=x_inputs_train.shape[1], kernel_regularizer=regularizers.l2(Lambda), kernel_initializer = initializers.glorot_uniform(seed=seed)))
    elif reg == 'l1':
        model.add(Dense(neurons, activation=act,input_dim=x_inputs_train.shape[1], kernel_regularizer=regularizers.l1(Lambda), kernel_initializer = initializers.glorot_uniform(seed=seed)))
    else:
        model.add(Dense(neurons, activation=act, input_dim=x_inputs_train.shape[1], kernel_regularizer=regularizers.l1_l2(l1 = Lambda, l2 = Lambda), kernel_initializer = initializers.glorot_uniform(seed=seed)))
    #model.add(Dropout(d1))
    #model.add(Dense(n2, activation='sigmoid', kernel_regularizer=regularizers.l2(Lambda)))
    if reg == 'l2':
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(Lambda), kernel_initializer = initializers.glorot_uniform(seed=seed)))
    elif reg == 'l1':
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(Lambda), kernel_initializer = initializers.glorot_uniform(seed=seed)))
    else:
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1 = Lambda, l2 = Lambda), kernel_initializer = initializers.glorot_uniform(seed=seed)))
    #model.add(Dropout(d2))
    if opt == 'sgd':
        optim = optimizers.SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = optimizers.Adam(lr=LR, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=surv_likelihood_loss, optimizer=optim)#sgd)#adam)#, metrics=['cindex_score']) #optimizers.RMSprop())
    return model

hn = np.int(np.ceil(np.sqrt(x_train.shape[1])))
Lambda = np.exp(np.arange(-6,2, 0.5))
BSLambda = [[10, Lambda[0]], [25, Lambda[0]], [5, Lambda[0]], [15, Lambda[0]], [20, Lambda[0]],
            [10, Lambda[1]], [25, Lambda[1]], [5, Lambda[1]], [15, Lambda[1]],[20, Lambda[1]],
            [10, Lambda[2]], [25, Lambda[2]], [5, Lambda[2]], [15, Lambda[2]],[20, Lambda[2]]]

n_train_valid = x_train.shape[0] + x_valid.shape[0]
print(n_train_valid)
n_train_valid_inputs = x_inputs_train.shape[0] + x_inputs_valid.shape[0]
print(n_train_valid_inputs)
#perm = np.arange(x_train_valid_2d.shape[0])
perm = np.arange(n_train_valid)
perm_vf = np.arange(n_train_valid_inputs)
perm_2d = perm*pas
size = np.int(n_train_valid/5)
size_vf = size*pas #np.int(n_train_valid_inputs/5)
scores_train = 0#np.zeros([len(hyprparams)])
scores_test = 0#np.zeros([len(hyperparams)])
ctda = []
ctde = []
cta = []
cte = []
dict_cv_model = {}
dict_cv_history = {}
dict_cv_cindex_train = {}
dict_cv_cindex_valid = {}
#di = np.concatenate(([0],np.cumsum(ind_time_train_valid[:-1])))
#print(di)
#fi = np.cumsum(ind_time_train_valid)
#print(fi)
#r0 = np.arange(0, ind_time_train_valid[0])
#rank_double = append(r0, [range(ind_time_train_valid[i], ind_time_train_valid[i+1]) for i in np.arange(len(ind_time_train_valid))])
#print(rank_double)
for k in np.arange(5):
    key = "k" + str(k)
    rank = perm[(k*size):((k+1)*size)]
    #print(rank)
    #rank_bis = [np.arange(di[i],fi[i]) for i in rank]
    #rank_bis2 = np.concatenate((rank_bis), axis = 0)
    rank_vf = perm_vf[(k*size_vf):((k+1)*size_vf)]
    #print(rank_bis)
    #print(rank_bis2)
    rank_vf = perm_vf[(k*size_vf):((k+1)*size_vf)]
    dataTrain = data_train_valid.drop(rank, axis = 0)
    x_train_NN = np.delete(x_inputs_train_valid, rank_vf, 0)
    #print(x_train_NN.shape)
    y_train_NN = np.delete(dil_train_valid_2d, rank_vf, 0)
    #print(y_train_NN)
    x_valid_NN = x_inputs_train_valid[rank_vf,:]
    #print(x_valid_NN.shape)
    y_valid_NN = dil_train_valid_2d[rank_vf,:]
    #print(y_valid_NN)
    dataValid = data_train_valid.loc[rank]
    model_cv = create_model('sigmoid', hn, BSLambda[h][1], 0.0, 0.0, 'l2', 0.0001, 'adam')
    ## pas de early stopping
    #early_stopping = EarlyStopping(monitor='val_loss', patience=2) #cindex_score
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights = True, verbose = 2, min_delta = 0.0002)
    history_cv = model_cv.fit(x_train_NN, y_train_NN, batch_size=BSLambda[h][0], epochs=500, validation_data = (x_valid_NN, y_valid_NN), callbacks=[early_stopping])
    dict_cv_model[key] = model_cv
    dict_cv_history[key] = history_cv.history
    ypred_train_NN = model_cv.predict_proba(x_train_NN)
    for layer in model_cv.layers:
        g=layer.get_config()
        w=layer.get_weights()
        print(w)
    get_target = lambda df: (df['time'].values, df['dead'].values)
    time_valid, dead_valid = get_target(dataValid)
    ypred_train_NN = model_cv.predict_proba(x_train_NN)
    ypred_test_NN = model_cv.predict_proba(x_valid_NN)
    ypred_surv_train_NN = ypred_train_NN.reshape([dataTrain.shape[0],-1])
    ypred_surv_valid_NN = ypred_test_NN.reshape([dataValid.shape[0],-1])
    y_pred_valid_surv = np.cumprod((1-ypred_surv_valid_NN), axis = 1)
    y_pred_train_surv = np.cumprod((1-ypred_surv_train_NN), axis = 1)
    oneyr_surv_train = y_pred_train_surv[:,50]
    oneyr_surv_valid = y_pred_valid_surv[:,50]
    surv_valid = pd.DataFrame(np.transpose(y_pred_valid_surv))
    surv_valid.index = interval_l
    surv_train = pd.DataFrame(np.transpose(y_pred_train_surv))
    surv_train.index = interval_l
    dict_cv_cindex_train[key] = concordance_index(dataTrain.time,oneyr_surv_train)
    dict_cv_cindex_valid[key] = concordance_index(dataValid.time,oneyr_surv_valid)
    #scores_train += concordance_index(dataTrain.time,oneyr_surv_train)#,data_train.dead)
    #scores_test += concordance_index(dataValid.time,oneyr_surv_valid)
    #cta.append(concordance_index(dataTrain.time,oneyr_surv_train))
    #cte.append(concordance_index(dataValid.time,oneyr_surv_valid))
    ev_valid = EvalSurv(surv_valid, time_valid, dead_valid, censor_surv='km')
    scores_test += ev_valid.concordance_td()
    ev_train = EvalSurv(surv_train, dataTrain['time'].values, dataTrain['dead'].values, censor_surv='km')
    scores_train += ev_train.concordance_td()
    cta.append(concordance_index(dataTrain.time,oneyr_surv_train))
    cte.append(concordance_index(dataValid.time,oneyr_surv_valid))
    ctda.append(ev_train.concordance_td())
    ctde.append(ev_valid.concordance_td())
    #scores_test += cindex_CV_score(Yvalid, ypred_test)
    #scores_train += cindex_CV_score(Ytrain, ypred_train)


save_loss_png = "output/loss_cv_" + str(h) + "_KIRC.png"
import seaborn as sns
fs = 20
plt.rc('axes', facecolor = "white", linewidth = 1,
       grid = False, edgecolor = "black", titlesize = fs, labelsize = fs)
plt.rc('font', size = fs)
plt.rc('xtick', labelsize = fs)
plt.rc('xtick', labelsize = fs)
plt.rc('ytick', labelsize = fs)
plt.rc('legend', fontsize = fs)
plt.rc('figure', titlesize = fs, figsize = (15, 10))

# A way to get different colors for each different index
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def get_color(i): return colors[np.mod(i, len(colors))]
keys = ["k" + str(_) for _ in np.arange(5)]
for i in range(0,len(keys)):
    plt.plot(dict_cv_history[keys[i]]['loss'],
             color = get_color(i), label = keys[i]+"_train")
    plt.plot(dict_cv_history[keys[i]]['val_loss'],
             linestyle = "--", color = get_color(i), label = keys[i]+"_valid")
loss_max = max([max(dict_cv_history[_]['loss']) for _ in keys])
val_loss_max = loss_max = max([max(dict_cv_history[_]['val_loss']) for _ in keys])
lmax = max(loss_max, val_loss_max)
plt.xlim(0, 499)
plt.ylim(0, lmax)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc = "best", fontsize = 10)
plt.savefig(save_loss_png)

scores_train /= 5
scores_test /= 5
#~ std_scores_train =
#scores_test = scores_test.reshape(len(hyperparams),-1)
#scores_train = scores_train.reshape(len(hyperparams),-1)
#cv_res = np.concatenate((Lambda[h],scores_train,scores_test), axis = None)
#ind_best = np.argmax(scores_test)
#best_model = hyperparams[ind_best]
#print("The best model is:", best_model, "Cindex_test", scores_test[ind_best], "cindex_train", scores_train[ind_best])
#cv_res_df = pd.DataFrame(cv_res)
#cv_res_file = "file_CCDix_CV_res_tab.csv"
#cv_res_df.to_csv(cv_res_file,index=False)

ci_cv_train = []
ci_cv_test = []
ci_cv_valid = []
cindex_max = 0.
best_model = "my_best_model_KIRC_" + str(h) + ".h5"
for k in np.arange(5):
    model = create_model('sigmoid', hn, BSLambda[h][1], 0.0, 0.0, 'l2', 0.0001, 'adam', seed = k)
    history=model.fit(x_inputs_train, dil_train_2d, batch_size=BSLambda[h][0], epochs=500, validation_data = (x_inputs_valid, dil_valid_2d))
    #for layer in model.layers:
    #    g=layer.get_config()
    #    w=layer.get_weights()
    #    print (g)
    #    print (w)
    y_pred=model.predict_proba(x_inputs_train,verbose=0)
    y_pred_surv = y_pred.reshape([data_train.shape[0],-1])
    y_pred_train_surv = np.cumprod((1-y_pred_surv), axis = 1)
    #n_test = x_test_sep.shape[0]
    oneyr_surv_train = y_pred_train_surv[:,50]
    y_pred_valid=model.predict_proba(x_inputs_valid,verbose=0)
    y_pred_surv_valid = y_pred_valid.reshape([data_valid.shape[0],-1])
    y_pred_valid_surv = np.cumprod((1-y_pred_surv_valid), axis = 1)
    #n_test = x_test_sep.shape[0]
    oneyr_surv_valid = y_pred_valid_surv[:,50]
    surv_valid = pd.DataFrame(np.transpose(y_pred_valid_surv))
    ev_actual = EvalSurv(surv_valid, data_valid['time'].values, data_valid['dead'].values, censor_surv='km')
    cindex_actual = ev_actual.concordance_td()
    if cindex_actual > cindex_max:
        print(k)
        cindex_max = cindex_actual
        model.save(best_model)
    surv_train = pd.DataFrame(np.transpose(y_pred_train_surv))
    ev_train = EvalSurv(surv_train, data_train['time'].values, data_train['dead'].values, censor_surv='km')
    ci_cv_train.append(ev_train.concordance_td())
    ci_cv_valid.append(cindex_actual)
    #perm = np.arange(n_test)
    #np.random.shuffle(perm)
    y_pred_test=model.predict_proba(x_inputs_test,verbose=0)
    y_pred_surv_test_sep = y_pred_test.reshape([data_test.shape[0],-1])
    y_pred_test_surv = np.cumprod((1-y_pred_surv_test_sep), axis = 1)#[np.cumprod(1-y_pred_surv_test_sep[i]) for i in np.arange(len(y_pred_surv_test_sep))]#np.cumprod(1-y_pred_test, axis
    print(y_pred_test_surv)
    print(y_pred_test_surv.shape)
    oneyr_surv_test = y_pred_test_surv[:,50]#[y_pred_test_surv[i][0] for i in np.arange(len(y_pred_test_surv))]#y_pred_test_surv[:,np.nonzero(g>1500)[0][0]]
    print(oneyr_surv_test)
    surv_test = pd.DataFrame(np.transpose(y_pred_test_surv))
    surv_test.index = interval_l
    #ci_cv_train.append(concordance_index(data_train.time,oneyr_surv_train))
    ev_test = EvalSurv(surv_test, data_test['time'].values, data_test['dead'].values, censor_surv='km')
    ci_cv_test.append(ev_test.concordance_td())

print("------best model--------")
del model  # deletes the existing model
model = load_model(best_model, compile=False)
y_pred=model.predict_proba(x_inputs_train,verbose=0)
y_pred_surv = y_pred.reshape([data_train.shape[0],-1])
y_pred_train_surv = np.cumprod((1-y_pred_surv), axis = 1)
oneyr_surv_train = y_pred_train_surv[:,50]
best_cindex_train = concordance_index(data_train.time,oneyr_surv_train)
surv_train = pd.DataFrame(np.transpose(y_pred_train_surv))
surv_train.index = interval_l
best_ctd_train = EvalSurv(surv_train, data_train['time'].values, data_train['dead'].values, censor_surv='km').concordance_td()

y_pred_valid=model.predict_proba(x_inputs_valid,verbose=0)
y_pred_surv_valid = y_pred_valid.reshape([data_valid.shape[0],-1])
y_pred_valid_surv = np.cumprod((1-y_pred_surv_valid), axis = 1)
oneyr_surv_valid = y_pred_valid_surv[:,50]
best_cindex_valid = concordance_index(data_valid.time,oneyr_surv_valid)
surv_valid = pd.DataFrame(np.transpose(y_pred_valid_surv))
surv_valid.index = interval_l
best_ctd_valid = EvalSurv(surv_valid, data_valid['time'].values, data_valid['dead'].values, censor_surv='km').concordance_td()

y_pred_test=model.predict_proba(x_inputs_test,verbose=0)
y_pred_surv_test = y_pred_test.reshape([data_test.shape[0],-1])
y_pred_test_surv = np.cumprod((1-y_pred_surv_test), axis = 1)
oneyr_surv_test = y_pred_test_surv[:,50]
best_cindex_test = concordance_index(data_test.time,oneyr_surv_test)
surv_test = pd.DataFrame(np.transpose(y_pred_test_surv))
surv_test.index = interval_l
best_ctd_test = EvalSurv(surv_test, data_test['time'].values, data_test['dead'].values, censor_surv='km').concordance_td()

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
print("---res Ctd---")
print(ci_cv_train)
print(ci_cv_valid)
print(ci_cv_test)
print("--------")
print("---res Cindex---")
print("Cindex_train : ", best_cindex_train)
print("Cindex_valid : ", best_cindex_valid)
print("Cindex_test : ", best_cindex_test)
print("---res Ctd---")
print("Ctd_train : ", best_ctd_train)
print("Ctd_valid : ", best_ctd_valid)
print("Ctd_test : ", best_ctd_test)

ci_train = np.mean(ci_cv_train)
ci_test = np.mean(ci_cv_test)

file = open('output/fichier_KIRC_vf.csv','a')
file.write(str(h) + "_10_2e-4," + 'sigmoid' + "," + str(hn) + "," + str(BSLambda[h][0]) + "," + str(BSLambda[h][1]) + ","  + "0.0" + "," + "0.0" + "," + "l2" + "," + "0.0001" + "," + "adam"+","+ str(scores_train) +","+ str(scores_test) + ","+ str(best_cindex_train) + "," + str(best_cindex_test) + "\n")
#'str(h) + "," + str(cv_res[0]) + "," + str(cv_res[1]) + ","+ str(cv_res[2]) + "," + str(cv_res[3]) + "," + str(cv_res[4]) + ","+ str(cv_res[5]) + ","+ str(cv_res[6]) + ","+ str(cv_res[7]) + ","+ str(scores_train) +","+ str(scores_test) + ","+ str(ci_train) + "," + str(ci_test) + "\n")
file.close()

## to add to NNsurv files on fusion
file_preds_train = "output/pred_files/preds_NNsurv_KIRC_vf_" + str(h) + ".csv"
#file_preds_test = "DATA/res/preds_NNsurv_test_CM_" + str(j) + ".csv"
file_preds_test = "output/pred_files/preds_NNsurv_KIRC_vf_" + str(h) + ".csv"
df_preds_NNsurv = pd.DataFrame(y_pred_test_surv)
df_preds_NNsurv_train = pd.DataFrame(y_pred_surv)
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

