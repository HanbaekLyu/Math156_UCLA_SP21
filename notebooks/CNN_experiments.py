import numpy as np
# from sklearn.utils.extmath import softmax
from matplotlib import pyplot as plt
import re
from tqdm import trange, tqdm
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from sklearn.datasets import fetch_openml

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

import sys
sys.path.append("../")
from src.CNN import CNN, compute_accuracy_metrics, multiclass_accuracy_metrics, list2onehot, onehot2list

def random_padding(img, thickness=1):
    # img = a x b image
    [a,b] = img.shape
    Y = np.zeros(shape=[a+thickness, b+thickness])
    r_loc = np.random.choice(np.arange(thickness+1))
    c_loc = np.random.choice(np.arange(thickness+1))
    Y[r_loc:r_loc+a, c_loc:c_loc+b] = img
    return Y


def sample_multiclass_MNIST_padding(list_digits=['0','1', '2'], full_MNIST=None, padding_thickness=10):
    # get train and test set from MNIST of given digits
    # e.g., list_digits = ['0', '1', '2']
    # pad each 28 x 28 image with zeros so that it has now "padding_thickness" more rows and columns
    # The original image is superimposed at a uniformly chosen location
    if full_MNIST is not None:
        X, y = full_MNIST
    else:
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        X = X / 255.
    Y = list2onehot(y.tolist(), list_digits)

    idx = [i for i in np.arange(len(y)) if y[i] in list_digits] # list of indices where the label y is in list_digits

    X01 = X[idx,:]
    y01 = Y[idx,:]

    X_train = []
    X_test = []
    y_test = [] # list of one-hot encodings (indicator vectors) of each label
    y_train = [] # list of one-hot encodings (indicator vectors) of each label

    for i in trange(X01.shape[0]):
        # for each example i, make it into train set with probabiliy 0.8 and into test set otherwise
        U = np.random.rand() # Uniform([0,1]) variable
        img_padded = random_padding(X01[i,:].reshape(28,28), thickness=padding_thickness)
        img_padded_vec = img_padded.reshape(1,-1)
        if U<0.8:
            X_train.append(img_padded_vec[0,:].copy())
            y_train.append(y01[i,:].copy())
        else:
            X_test.append(img_padded_vec[0,:].copy())
            y_test.append(y01[i,:].copy())

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    return X_train, X_test, y_train, y_test


# compute comparative multiclass classification metrics on test data

def main():
    padding_list = [0, 7, 13, 20]
    list_digits=['0','1','2','3','4']

    ## Train
    train_size_list = [50, 100, 200]

    # make plot
    ncols = len(train_size_list)
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=[13,5])

    for t in trange(len(train_size_list)):
        accuracy_list_test = []
        accuracy_list_train = []

        train_size = train_size_list[t]

        for thickness in tqdm(padding_list):
            # Data preprocessing
            X_train, X_test, y_train, y_test = sample_multiclass_MNIST_padding(list_digits=list_digits,
                                                                               full_MNIST=None,
                                                                               padding_thickness=thickness)

            idx = np.random.choice(np.arange(len(y_train)), train_size)
            X_train0 = X_train[idx, :]/np.max(X_train)
            y_train0 = y_train[idx, :]



            idx = np.random.choice(np.arange(len(y_test)), 3000)
            X_test0 = X_test[idx, :]/np.max(X_test)
            y_test0 = y_test[idx, :]

            out = []
            out_train = []
            # populate the tuple list with the data
            for i in range(X_train0.shape[0]):
                item = list((X_train0[i,:].reshape(1, 28+thickness, 28+thickness), y_train0[i,:]))

                out.append(item)
                out_train.append(X_train0[i,:].reshape(1, 28+thickness, 28+thickness))

            X_test /= np.max(X_test)
            out_test = []
            for i in range(X_test0.shape[0]):
                out_test.append((X_test0[i,:].reshape(1, 28+thickness, 28+thickness)))



            # FFNN training
            CNN0 = CNN(training_data = out,
               f = 5, # conv filter dim
               f_pool = 2, # maxpool filter dim
               num_filt1 = 10, # num filters for the first conv layer
               num_filt2 = 10, # num filters for the second conv layer
               conv_stride = 1,
               pool_stride = 2,
               hidden_nodes = 128)

            CNN0.train(lr = 0.01,
                       beta1 = 0.95,
                       beta2 = 0.99,
                       minibatch_size = 32,
                       num_epochs = 500,
                       verbose = True)

            # FFNN prediction
            print()
            y_hat_train = np.asarray(CNN0.predict(out_train))
            y_hat_test = np.asarray(CNN0.predict(out_test))

            y_train_label = np.asarray(onehot2list(y_train0))
            y_test_label = np.asarray(onehot2list(y_test0))

            results_train = multiclass_accuracy_metrics(Y_test=y_train0, P_pred=y_hat_train)
            results_test = multiclass_accuracy_metrics(Y_test=y_test0, P_pred=y_hat_test)

            accuracy_list_train.append(results_train.get('Accuracy'))
            accuracy_list_test.append(results_test.get('Accuracy'))

        ## Plot
        ax[t].plot(padding_list, accuracy_list_train, color='blue', label="train accuracy")
        ax[t].plot(padding_list, accuracy_list_test, color='red', label="test accuracy")
        ax[t].set_xlabel('Padding thickness', fontsize=15)
        ax[t].set_ylabel('Classification Accuracy', fontsize=15)
        ax[t].title.set_text("num training ex = %i" % (train_size))
        ax[t].legend(fontsize=15)

    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.savefig('MNIST_CNN_accuracy_padding_ex2.pdf')


main()
