from models.moment import MOMENTPipeline


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pickle,sys,h5py
import keras
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.regularizers import *


def to_amp_phase(X_train, X_val, X_test, nsamples):
    X_train_cmplx = X_train[:, :, 0] + 1j * X_train[:, :, 1]
    X_val_cmplx = X_val[:, :, 0] + 1j * X_val[:, :, 1]
    X_test_cmplx = X_test[:, :, 0] + 1j * X_test[:, :, 1]

    X_train_amp = np.abs(X_train_cmplx)
    X_train_ang = np.arctan2(X_train[:, :, 1], X_train[:, :, 0]) / np.pi

    X_train_amp = np.reshape(X_train_amp, (-1, 1, nsamples))
    X_train_ang = np.reshape(X_train_ang, (-1, 1, nsamples))

    X_train = np.concatenate((X_train_amp, X_train_ang), axis=1)
    X_train = np.transpose(np.array(X_train), (0, 2, 1))

    X_val_amp = np.abs(X_val_cmplx)
    X_val_ang = np.arctan2(X_val[:, :, 1], X_val[:, :, 0]) / np.pi

    X_val_amp = np.reshape(X_val_amp, (-1, 1, nsamples))
    X_val_ang = np.reshape(X_val_ang, (-1, 1, nsamples))

    X_val = np.concatenate((X_val_amp, X_val_ang), axis=1)
    X_val = np.transpose(np.array(X_val), (0, 2, 1))

    X_test_amp = np.abs(X_test_cmplx)
    X_test_ang = np.arctan2(X_test[:, :, 1], X_test[:, :, 0]) / np.pi

    X_test_amp = np.reshape(X_test_amp, (-1, 1, nsamples))
    X_test_ang = np.reshape(X_test_ang, (-1, 1, nsamples))

    X_test = np.concatenate((X_test_amp, X_test_ang), axis=1)
    X_test = np.transpose(np.array(X_test), (0, 2, 1))
    return (X_train, X_val, X_test)

RML2018_classes = ['OOK','4ASK','8ASK',
               'BPSK','QPSK','8PSK','16PSK','32PSK',
               '16APSK','32APSK','64APSK','128APSK',
               '16QAM','32QAM','64QAM','128QAM','256QAM',
               'AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC',
               'FM','GMSK','OQPSK']

def get_RNL2018_data(RNL2018_file=""):
    # def train(from_filename = '/media/norm_XYZ_1024_128k.hdf5',weight_file='weights/norm_res-like-128k.wts.h5',init_weight_file=None):
    from_filename = RNL2018_file
    f = h5py.File(from_filename, 'r')  # 打开h5文件
    X = f['X'][:, :, :]  # ndarray(2555904*1024*2)
    Y = f['Y'][:, :]  # ndarray(2M*24)
    Z = f['Z'][:]  # ndarray(2M*1)
    # [N,1024,2]
    in_shp = X[0].shape
    n_examples = X.shape[0]
    n_train = int(n_examples * 0.6)
    n_val = int(n_examples * 0.2)
    train_idx = list(np.random.choice(range(0, n_examples), size=n_train, replace=False))
    val_idx = list(np.random.choice(list(set(range(0, n_examples)) - set(train_idx)), size=n_val, replace=False))
    test_idx = list(set(range(0, n_examples)) - set(train_idx) - set(val_idx))
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    X_train = X[train_idx]
    Y_train = Y[train_idx]
    Z_train = Z[train_idx]

    X_val = X[val_idx]
    Y_val = Y[val_idx]
    Z_val = Z[val_idx]

    X_test = X[test_idx]
    Y_test = Y[test_idx]
    Z_test = Z[test_idx]
    # X_train, X_val, X_test = to_amp_phase(X_train, X_val, X_test, 1024)

    print(f"X_train:{X_train.shape}---Y_train:{Y_train.shape}---X_val:{X_val.shape}---Y_val:{Y_val.shape}---X_test:{X_test.shape}---Y_test:{Y_test.shape}")
    print(f"Z_train:{Z_train.shape}---Z_val:{Z_val.shape}---Z_test:{Z_test.shape}")

    return {X_train,Y_train,Z_train, X_val,Y_val,Z_val, X_test,Y_test,Z_test}


if __name__ == "__main__":
    model = MOMENTPipeline.from_pretrained("/user_home/WirelessData/AutonLab/MOMENT-1-large",
                                            model_kwargs={
                                                'task_name': 'classification',
                                                'n_channels': 4,
                                                'num_class': 24
                                            },)
    # takes in tensor of shape [batchsize, n_channels, context_length]
    x = torch.randn(2, 4, 512)
    output = model(x)
    pprint(output)

    logits = output.logits

    # [batch_size, ]
    predicted_labels = logits.argmax(dim=1)


    RML2018_dataset = get_RNL2018_data()
