'''
Description: Convolutional Neural Network implemented using numpy
Author: Alejandro Escontrela
Version: V.1.
Date: June 12th, 2018
Modification by Hanbaek Lyu (5/11/2021)
'''

import numpy as np
import pickle
from tqdm import tqdm, trange
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#####################################################
############### Building The Network ################
#####################################################

class CNN(object):
    """
    Input data type: training_data = [pattern1, pattern2, ..., pattern n]
    pattern i = [np.array (input), np.array (output in one-hot encoding)]
    Input --> Conv1 + ReLU + MaxPool --> Conv2 + ReLU + MaxPool --> Dense1 + ReLU --> Dense2 + Softmax --> predictive PMF
    """
    def __init__(self,
                 training_data,
                 f = 5, # conv filter dim
                 f_pool = 2, # maxpool filter dim
                 num_filt1 = 8, # num filters for the first conv layer
                 num_filt2 = 8, # num filters for the second conv layer
                 conv_stride = 1,
                 pool_stride = 2,
                 hidden_nodes = 128):

        self.training_data = training_data
        self.img_depth, self.img_x_dim, self.img_y_dim = training_data[0][0].shape
        print('self.img_x_dim', self.img_x_dim)
        self.num_classes = len(training_data[0][1])
        self.num_filt1 = num_filt1
        self.num_filt2 = num_filt2
        self.f = f # conv filter dim
        self.f_pool = f_pool # maxpool filter dim
        self.num_filt1 = num_filt1 # num filters for the first conv layer
        self.num_filt2 = num_filt2 # num filters for the second conv layer
        self.conv_stride = conv_stride
        self.pool_stride = pool_stride
        self.hidden_nodes = hidden_nodes


        ## Initializing all the parameters
        h_dim, w_dim = self.compute_conv_dim()
        f1 = self.initializeFilter((num_filt1 ,self.img_depth,f,f))
        f2 = self.initializeFilter((num_filt2 ,num_filt1,f,f)) # Initialize a set of filters for each channel (maybe unnecessary)
        w3 = self.initializeWeight((hidden_nodes, num_filt2 * h_dim * w_dim))
        w4 = self.initializeWeight((self.num_classes, hidden_nodes))

        # biases
        b1 = np.zeros((f1.shape[0],1))
        b2 = np.zeros((f2.shape[0],1))
        b3 = np.zeros((w3.shape[0],1))
        b4 = np.zeros((w4.shape[0],1))

        self.params = {}
        self.params.update({'f1': f1})
        self.params.update({'f2': f2})
        self.params.update({'w3': w3})
        self.params.update({'w4': w4})
        self.params.update({'b1': b1})
        self.params.update({'b2': b2})
        self.params.update({'b3': b3})
        self.params.update({'b4': b4})

    def compute_conv_dim(self):
        h_dim = self.img_x_dim
        w_dim = self.img_y_dim

        # conv1
        h_dim = int((h_dim - self.f)/self.conv_stride)+1
        w_dim = int((w_dim - self.f)/self.conv_stride)+1

        # maxpool
        h_dim = int((h_dim - self.f_pool)/self.pool_stride)+1
        w_dim = int((w_dim - self.f_pool)/self.pool_stride)+1

        # conv 2
        h_dim = int((h_dim - self.f)/self.conv_stride)+1
        w_dim = int((w_dim - self.f)/self.conv_stride)+1

        # maxpool
        h_dim = int((h_dim - self.f_pool)/self.pool_stride)+1
        w_dim = int((w_dim - self.f_pool)/self.pool_stride)+1
        return h_dim, w_dim

    def initializeFilter(self, size, scale = 1.0):
        stddev = scale/np.sqrt(np.prod(size))
        return np.random.normal(loc = 0, scale = stddev, size = size)

    def initializeWeight(self, size):
        return np.random.standard_normal(size=size) * 0.01

    def conv(self, image, label):

        # ForwardPropragate + BackPropagate + Compute Gradients

        conv_s, pool_f, pool_s = self.conv_stride, self.f_pool, self.pool_stride

        f1 = self.params.get('f1')
        f2 = self.params.get('f2')
        w3 = self.params.get('w3')
        w4 = self.params.get('w4')
        b1 = self.params.get('b1')
        b2 = self.params.get('b2')
        b3 = self.params.get('b3')
        b4 = self.params.get('b4')

        ################################################
        ############## Forward Operation ###############
        ################################################
        conv1 = self.convolution(image, f1, b1, conv_s) # convolution operation
        conv1[conv1<=0] = 0 # pass through ReLU non-linearity
        pooled1 = self.maxpool(conv1, pool_f, pool_s) # maxpooling operation

        conv2 = self.convolution(pooled1, f2, b2, conv_s) # second convolution operation
        conv2[conv2<=0] = 0 # pass through ReLU non-linearity
        pooled2 = self.maxpool(conv2, pool_f, pool_s) # maxpooling operation

        (nf2, dim2, _) = pooled2.shape
        fc = pooled2.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer
        # print('fc.shape', fc.shape)

        z = w3.dot(fc) + b3 # first dense layer
        z[z<=0] = 0 # pass through ReLU non-linearity
        # print('z.shape', z.shape)

        out = w4.dot(z) + b4 # second dense layer

        # print('out.shape', out.shape)

        probs = softmax(out) # predict class probabilities with the softmax activation function

        # print('probs.shape', probs.shape)

        ################################################
        #################### Loss ######################
        ################################################

        loss = categoricalCrossEntropy(probs, label) # categorical cross-entropy loss
        # print('loss',loss)

        ################################################
        ############# Backward Operation ###############
        ################################################

        dout = probs - label # derivative of loss w.r.t. final dense layer output

        dw4 = dout.dot(z.T) # loss gradient of final dense layer weights
        db4 = np.sum(dout, axis = 1).reshape(b4.shape) # loss gradient of final dense layer biases

        dz = w4.T.dot(dout) # loss gradient of first dense layer outputs
        dz[z<=0] = 0 # backpropagate through ReLU
        dw3 = dz.dot(fc.T)
        db3 = np.sum(dz, axis = 1).reshape(b3.shape)

        dfc = w3.T.dot(dz) # loss gradients of fully-connected layer (pooling layer)
        dpool2 = dfc.reshape(pooled2.shape) # reshape fully connected into dimensions of pooling layer

        dconv2 = self.maxpoolBackward(dpool2, conv2, pool_f, pool_s) # backprop through the max-pooling layer(only neurons with highest activation in window get updated)
        dconv2[conv2<=0] = 0 # backpropagate through ReLU
        dpool1, df2, db2 = self.convolutionBackward(dconv2, pooled1, f2, conv_s) # backpropagate previous gradient through second convolutional layer.

        dconv1 = self.maxpoolBackward(dpool1, conv1, pool_f, pool_s) # backprop through the max-pooling layer(only neurons with highest activation in window get updated)
        dconv1[conv1<=0] = 0 # backpropagate through ReLU
        dimage, df1, db1 = self.convolutionBackward(dconv1, image, f1, conv_s) # backpropagate previous gradient through first convolutional layer.

        grads = [df1, df2, dw3, dw4, db1, db2, db3, db4]

        return grads, loss



    #####################################################
    ################### Optimization ####################
    #####################################################

    def adamGD(self, minibatch, lr, beta1, beta2, cost):
        '''
        update the parameters through Adam gradient descnet (A fancy version of GD)
        batch = [pattern1, pattern2, .. pattern n']
        pattern i = [np.array (input img), np.array (output)]
        '''
        f1 = self.params.get('f1')
        f2 = self.params.get('f2')
        w3 = self.params.get('w3')
        w4 = self.params.get('w4')
        b1 = self.params.get('b1')
        b2 = self.params.get('b2')
        b3 = self.params.get('b3')
        b4 = self.params.get('b4')

        batch_size = len(minibatch)

        cost_ = 0

        # initialize gradients and momentum,RMS params
        # TODO: Simplify the following initialization
        df1 = np.zeros(f1.shape)
        df2 = np.zeros(f2.shape)
        dw3 = np.zeros(w3.shape)
        dw4 = np.zeros(w4.shape)
        db1 = np.zeros(b1.shape)
        db2 = np.zeros(b2.shape)
        db3 = np.zeros(b3.shape)
        db4 = np.zeros(b4.shape)

        v1 = np.zeros(f1.shape)
        v2 = np.zeros(f2.shape)
        v3 = np.zeros(w3.shape)
        v4 = np.zeros(w4.shape)
        bv1 = np.zeros(b1.shape)
        bv2 = np.zeros(b2.shape)
        bv3 = np.zeros(b3.shape)
        bv4 = np.zeros(b4.shape)

        s1 = np.zeros(f1.shape)
        s2 = np.zeros(f2.shape)
        s3 = np.zeros(w3.shape)
        s4 = np.zeros(w4.shape)
        bs1 = np.zeros(b1.shape)
        bs2 = np.zeros(b2.shape)
        bs3 = np.zeros(b3.shape)
        bs4 = np.zeros(b4.shape)

        for i in np.arange(len(minibatch)):
            x = minibatch[i][0].reshape(self.img_depth, self.img_x_dim, self.img_y_dim)
            y = minibatch[i][1].reshape(-1,1)

            # Collect Gradients for training example
            # stride for conv = 1
            # stride for maxpool = 2
            # maxpool filter dim = 2
            grads, loss = self.conv(image=x, label=y)
            [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = grads

            df1+=df1_
            db1+=db1_
            df2+=df2_
            db2+=db2_
            dw3+=dw3_
            db3+=db3_
            dw4+=dw4_
            db4+=db4_

            cost_+= loss

        # Parameter Update
        v1 = beta1*v1 + (1-beta1)*df1/batch_size # momentum update
        s1 = beta2*s1 + (1-beta2)*(df1/batch_size)**2 # RMSProp update
        f1 -= lr * v1/np.sqrt(s1+1e-7) # combine momentum and RMSProp to perform update with Adam

        bv1 = beta1*bv1 + (1-beta1)*db1/batch_size
        bs1 = beta2*bs1 + (1-beta2)*(db1/batch_size)**2
        b1 -= lr * bv1/np.sqrt(bs1+1e-7)

        v2 = beta1*v2 + (1-beta1)*df2/batch_size
        s2 = beta2*s2 + (1-beta2)*(df2/batch_size)**2
        f2 -= lr * v2/np.sqrt(s2+1e-7)

        bv2 = beta1*bv2 + (1-beta1) * db2/batch_size
        bs2 = beta2*bs2 + (1-beta2)*(db2/batch_size)**2
        b2 -= lr * bv2/np.sqrt(bs2+1e-7)

        v3 = beta1*v3 + (1-beta1) * dw3/batch_size
        s3 = beta2*s3 + (1-beta2)*(dw3/batch_size)**2
        w3 -= lr * v3/np.sqrt(s3+1e-7)

        bv3 = beta1*bv3 + (1-beta1) * db3/batch_size
        bs3 = beta2*bs3 + (1-beta2)*(db3/batch_size)**2
        b3 -= lr * bv3/np.sqrt(bs3+1e-7)

        v4 = beta1*v4 + (1-beta1) * dw4/batch_size
        s4 = beta2*s4 + (1-beta2)*(dw4/batch_size)**2
        w4 -= lr * v4 / np.sqrt(s4+1e-7)

        bv4 = beta1*bv4 + (1-beta1)*db4/batch_size
        bs4 = beta2*bs4 + (1-beta2)*(db4/batch_size)**2
        b4 -= lr * bv4 / np.sqrt(bs4+1e-7)


        cost_ = cost_/batch_size
        cost.append(cost_)

        params = {}
        params.update({'f1': f1})
        params.update({'f2': f2})
        params.update({'w3': w3})
        params.update({'w4': w4})
        params.update({'b1': b1})
        params.update({'b2': b2})
        params.update({'b3': b3})
        params.update({'b4': b4})

        return params, cost

    #####################################################
    ##################### Training ######################
    #####################################################

    def train(self,
              lr = 0.01,
              beta1 = 0.95,
              beta2 = 0.99,
              minibatch_size = 32,
              num_epochs = 2,
              verbose = False,
              save_path = 'params.pkl'):


        # np.random.shuffle(X)

        cost = []

        print("LR:"+str(lr)+", MiniBatch Size:"+str(minibatch_size))

        for epoch in trange(num_epochs):

            # sample a minibatch:
            #X = []
            #Y = []
            # idx = np.random.choice(np.arange(len(self.training_data)), minibatch_size)
            #pattern = self.training_data[idx[i]]
            #X.append(pattern[0])
            #Y.append(pattern[1])

            np.random.shuffle(self.training_data)
            batches = [self.training_data[k:k + minibatch_size] for k in range(0, len(self.training_data), minibatch_size)]
            for i in np.arange(len(batches)):
                params, cost = self.adamGD(minibatch=batches[i],
                                           lr=lr, # learning rate
                                           beta1=beta1,
                                           beta2=beta2,
                                           cost=cost)
                # t.set_description("Cost: %.2f" % (cost[-1]))
            # update parameters

            self.params = params
            if (epoch % 5 == 0) and verbose:
                print('epoch %i, error %-.5f' % (epoch, cost[-1]))

            #if cost[-1] < 0.01:
            #    break

        to_save = [params, cost]


        with open(save_path, 'wb') as file:
            pickle.dump(to_save, file)

        return cost



    #####################################################
    ################ Forward Operations #################
    #####################################################


    def convolution(self, image, filt, bias, s=1):
        '''
        Confolves `filt` over `image` using stride `s`
        '''
        (n_f, n_c_f, f, _) = filt.shape # filter dimensions (# filters, filter depth, filter dim, filder dim)
        # print('filt.shape', filt.shape)
        n_c, in_dim, _ = image.shape # image dimensions
        # print('image.shape', image.shape)

        out_dim = int((in_dim - f)/s)+1 # calculate output dimensions

        assert n_c == n_c_f, "Filter depth must match depth of input image"

        out = np.zeros((n_f,out_dim,out_dim))

        # convolve the filter over every part of the image, adding the bias at each step.
        for curr_f in range(n_f):
            curr_y = out_y = 0
            while curr_y + f <= in_dim:
                curr_x = out_x = 0
                while curr_x + f <= in_dim:
                    out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f]) + bias[curr_f]
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1

        return out

    def maxpool(self, image, f=2, s=2):
        '''
        Downsample `image` using kernel size `f` and stride `s`
        '''
        n_c, h_prev, w_prev = image.shape

        h = int((h_prev - f)/s)+1
        w = int((w_prev - f)/s)+1

        downsampled = np.zeros((n_c, h, w))
        for i in range(n_c):
            # slide maxpool window over each part of the image and assign the max value at each step to the output
            curr_y = out_y = 0
            while curr_y + f <= h_prev:
                curr_x = out_x = 0
                while curr_x + f <= w_prev:
                    downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
        return downsampled




    #####################################################
    ############### Backward Operations #################
    #####################################################

    def convolutionBackward(self, dconv_prev, conv_in, filt, s):
        '''
        Backpropagation through a convolutional layer.
        '''
        (n_f, n_c, f, _) = filt.shape
        (_, orig_dim, _) = conv_in.shape
        ## initialize derivatives
        dout = np.zeros(conv_in.shape)
        dfilt = np.zeros(filt.shape)
        dbias = np.zeros((n_f,1))
        for curr_f in range(n_f):
            # loop through all filters
            curr_y = out_y = 0
            while curr_y + f <= orig_dim:
                curr_x = out_x = 0
                while curr_x + f <= orig_dim:
                    # loss gradient of filter (used to update the filter)
                    dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y+f, curr_x:curr_x+f]
                    # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                    dout[:, curr_y:curr_y+f, curr_x:curr_x+f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f]
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
            # loss gradient of the bias
            dbias[curr_f] = np.sum(dconv_prev[curr_f])

        return dout, dfilt, dbias

    def maxpoolBackward(self, dpool, orig, f, s):
        '''
        Backpropagation through a maxpooling layer. The gradients are passed through the indices of greatest value in the original maxpooling during the forward step.
        '''
        (n_c, orig_dim, _) = orig.shape

        dout = np.zeros(orig.shape)

        for curr_c in range(n_c):
            curr_y = out_y = 0
            while curr_y + f <= orig_dim:
                curr_x = out_x = 0
                while curr_x + f <= orig_dim:
                    # obtain index of largest value in input for current window
                    (a, b) = nanargmax(orig[curr_c, curr_y:curr_y+f, curr_x:curr_x+f])
                    dout[curr_c, curr_y+a, curr_x+b] = dpool[curr_c, out_y, out_x]

                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1

        return dout


    #####################################################
    ##################### Prediction ####################
    #####################################################

    def predict(self, image_list):
        '''
        Make predictions with trained filters/weights.
        '''
        conv_s, pool_f, pool_s = self.conv_stride, self.f_pool, self.pool_stride
        f1 = self.params.get('f1')
        f2 = self.params.get('f2')
        w3 = self.params.get('w3')
        w4 = self.params.get('w4')
        b1 = self.params.get('b1')
        b2 = self.params.get('b2')
        b3 = self.params.get('b3')
        b4 = self.params.get('b4')

        list_probs = []

        for i in trange(len(image_list)):
            image = image_list[i]
            conv1 = self.convolution(image, f1, b1, conv_s) # convolution operation
            conv1[conv1<=0] = 0 #relu activation
            pooled1 = self.maxpool(conv1, pool_f, pool_s) # maxpooling operation

            conv2 = self.convolution(pooled1, f2, b2, conv_s) # second convolution operation
            conv2[conv2<=0] = 0 # pass through ReLU non-linearity
            pooled = self.maxpool(conv2, pool_f, pool_s) # maxpooling operation
            (nf2, dim2, _) = pooled.shape

            fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer

            z = w3.dot(fc) + b3 # first dense layer
            z[z<=0] = 0 # pass through ReLU non-linearity

            out = w4.dot(z) + b4 # second dense layer
            probs = softmax(out) # predict class probabilities with the softmax activation function
            list_probs.append(probs)

        return list_probs

#####################################################
################## Utility Methods ##################
#####################################################

def softmax(X):
    out = np.exp(X)
    return out/np.sum(out)

def categoricalCrossEntropy(probs, label):
    return -np.sum(label * np.log(probs))

def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs

def compute_accuracy_metrics(Y_test, P_pred, use_opt_threshold=False, verbose=True):

    # y_test = binary label
    # P_pred = predicted probability for y_test
    # compuate various binary classification accuracy metrics
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, P_pred, pos_label=None)
    mythre = thresholds[np.argmax(tpr - fpr)]
    myauc = metrics.auc(fpr, tpr)
    # print('!!! auc', myauc)

    # Compute classification statistics
    threshold = 0.5
    if use_opt_threshold:
        threshold = mythre

    Y_pred = P_pred.copy()
    Y_pred[Y_pred < threshold] = 0
    Y_pred[Y_pred >= threshold] = 1

    mcm = confusion_matrix(Y_test, Y_pred)

    tn = mcm[0, 0]
    tp = mcm[1, 1]
    fn = mcm[1, 0]
    fp = mcm[0, 1]

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tn / (tn + fp)
    specificity = tp / (tp + fn)
    precision = tp / (tp + fp)
    fall_out = fp / (fp + tn)
    miss_rate = fn / (fn + tp)

    # Save results
    results_dict = {}
    results_dict.update({'Y_test': Y_test})
    results_dict.update({'Y_pred': Y_pred})
    results_dict.update({'AUC': myauc})
    results_dict.update({'Opt_threshold': mythre})
    results_dict.update({'Accuracy': accuracy})
    results_dict.update({'Sensitivity': sensitivity})
    results_dict.update({'Specificity': specificity})
    results_dict.update({'Precision': precision})
    results_dict.update({'Fall_out': fall_out})
    results_dict.update({'Miss_rate': miss_rate})
    results_dict.update({'Confusion_mx': mcm})


    if verbose:
        for key in [key for key in results_dict.keys()]:
            if key not in ['Y_test', 'Y_pred', 'Confusion_mx']:
                print('% s ===> %.3f' % (key, results_dict.get(key)))
        print('Confusion matrix  ===> \n', mcm)

    return results_dict


def multiclass_accuracy_metrics(Y_test, P_pred, class_labels=None, use_opt_threshold=False):
    # y_test = multiclass one-hot encoding  labels [samples x labels]
    # Q = predicted probability for y_test
    # compuate various classification accuracy metrics
    results_dict = {}
    y_test = []
    y_pred = []
    for i in np.arange(Y_test.shape[0]):
        for j in np.arange(Y_test.shape[1]):
            if Y_test[i,j] == 1:
                y_test.append(j)
            if P_pred[i,j] == np.max(P_pred[i,:]):
                # print('!!!', np.where(P_pred[i,:]==np.max(P_pred[i,:])))
                y_pred.append(j)

    confusion_mx = metrics.confusion_matrix(y_test, y_pred)
    results_dict.update({'confusion_mx':confusion_mx})
    results_dict.update({'Accuracy':np.trace(confusion_mx)/np.sum(np.sum(confusion_mx))})
    print('!!! confusion_mx', confusion_mx)
    print('!!! Accuracy', results_dict.get('Accuracy'))


    return results_dict

def list2onehot(y, list_classes):
    """
    y = list of class lables of length n
    output = n x k array, i th row = one-hot encoding of y[i] (e.g., [0,0,1,0,0])
    """
    Y = np.zeros(shape = [len(y), len(list_classes)], dtype=int)
    for i in np.arange(Y.shape[0]):
        for j in np.arange(len(list_classes)):
            if y[i] == list_classes[j]:
                Y[i,j] = 1
    return Y

def onehot2list(y, list_classes=None):
    """
    y = n x k array, i th row = one-hot encoding of y[i] (e.g., [0,0,1,0,0])
    output =  list of class lables of length n
    """
    if list_classes is None:
        list_classes = np.arange(y.shape[1])

    y_list = []
    for i in np.arange(y.shape[0]):
        idx = np.where(y[i,:]==1)
        idx = idx[0][0]
        y_list.append(list_classes[idx])
    return y_list
