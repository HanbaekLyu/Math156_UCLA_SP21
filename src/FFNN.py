import numpy as np

'''
Feedforward Neural Network solely based on numpy.
Uses SGD for training and backpropagation for gradient evaluation.
Author: Hanbaek Lyu (5/3/2021)
'''

class FFNN():

    def __init__(self,
                 list_hidden_layer_sizes = [30], # hidden1, hidden2, .. , hidden h
                 loss_function = 'softmax-cross-entropy', # or 'square' or 'cross-entropy'
                 activation_list = None, # ['ReLU', 'sigmoid'],
                 node_states = None,
                 weight_matrices = None,
                 training_set = [None, None]): # input = [feature_dim x samples], output [\kappa x samples]

        self.training_set = training_set
        # self.test_set = test_set
        self.list_layer_sizes = [self.training_set[0].shape[0]] + list_hidden_layer_sizes + [self.training_set[1].shape[0]]
        self.loss_function = loss_function
        self.n_layers = len(self.list_layer_sizes)-1
        self.activation_list = activation_list
        self.node_states = node_states
        self.weight_matrices = weight_matrices

        self.initialize()


    def initialize(self):
        if self.activation_list is None:
            activation_list = ['ReLU' for i in np.arange(len(self.list_layer_sizes)-1)]
            activation_list[-1] = 'sigmoid'
            self.activation_list = activation_list

        if self.node_states == None:
            node_states = []
            for i in np.arange(len(self.list_layer_sizes)):
                node_states.append(np.zeros(shape=[self.list_layer_sizes[i], ]))
            self.node_states = node_states

        if self.weight_matrices == None:
            weight_matrices = []
            for i in np.arange(len(self.activation_list)):
                U = np.random.rand(self.list_layer_sizes[i], self.list_layer_sizes[i+1])
                weight_matrices.append(1-2*U)
            self.weight_matrices = weight_matrices


    def forward_propagate(self, input_data):
        # Forward propagate the input using the current weights and update node states
        self.node_states[0] = input_data
        for i in np.arange(self.n_layers):
            X_new = self.node_states[i].T @ self.weight_matrices[i]
            X_new = activation(X_new, type=self.activation_list[i])
            self.node_states[i+1] = X_new
            # print('!!! X_new', X_new)

    def backpropagate(self, output_data):
        # Backpropagate the error and return the gradient of the weight matrices
        # output_data = column array
        node_errors = self.node_states.copy()

        y = output_data
        y_hat = self.node_states[-1] # shape (\kappa, )
        y_hat = y_hat[:, np.newaxis]
        node_errors[-1] = delta_loss_function(y=y, y_hat=y_hat, type=self.loss_function)
        W_grad = self.weight_matrices.copy()

        for i in range(self.n_layers -1, -1, -1):
            # First weight the errors of nodes in layer above by the derivative of activation
            wtd_errors = node_errors[i+1].copy()
            z = self.node_states[i][:,np.newaxis]
            W = self.weight_matrices[i]
            layer_size_above = self.list_layer_sizes[i+1]
            delta_activation_weights = [delta_activation(z.T @ W[:,q], type=self.activation_list[i]) for q in np.arange(layer_size_above)]
            delta_activation_weights = np.asarray(delta_activation_weights)
            if len(delta_activation_weights.shape)==1:
                delta_activation_weights = delta_activation_weights[:,np.newaxis]
            if len(wtd_errors.shape)==1:
                wtd_errors = wtd_errors[:,np.newaxis]

            wtd_errors = wtd_errors * delta_activation_weights
            # wtd_errors = wtd_errors[:, np.newaxis]

            # Compute the gradient of the i th weight matrix (conneting layer i and i+1)
            W_grad[i] = wtd_errors @ z.T

            # Propagate it backward onto layer i
            node_errors[i] = (W @ wtd_errors)[:,0]
        return W_grad

    def minibatch_grad(self, minibatch_idx):

        W_grad_list = []
        Y = self.training_set[1] # true labels: each column = one-hot encoding
        X = self.training_set[0]
        for i in minibatch_idx:
            self.forward_propagate(input_data=X[:,i])
            y = Y[:, i]
            y = y[:, np.newaxis]
            W_grad = self.backpropagate(output_data=y)
            W_grad_list.append(W_grad)

        W_grad_minibatch = self.weight_matrices.copy()
        for j in np.arange(self.n_layers):
            grad_temp = [W_grad_list[i][j] for i in np.arange(len(minibatch_idx))]
            W_grad_minibatch[j] = np.sum(np.asarray(grad_temp), axis=0)

        return W_grad_minibatch

    def train(self, n_SGD_iter=10, minibatch_size=1, stopping_diff=0.01):
        Y = self.training_set[1]
        for i in np.arange(n_SGD_iter):
            # compute the minibatch gradients of weight matrices
            num_train_data = Y.shape[0]
            minibatch_idx = np.random.choice(np.arange(num_train_data), minibatch_size)
            W_grad_minibatch = self.minibatch_grad(minibatch_idx=minibatch_idx)

            # GD
            for j in np.arange(self.n_layers):
                W1 = self.weight_matrices[j]
                t = 0
                grad = W_grad_minibatch[j].T
                if (np.linalg.norm(grad) > stopping_diff):
                    W1 = W1 - (np.log(i+1) / (((i + 1) ** (0.5)))) * grad
                self.weight_matrices[j] = W1.copy()
                if j == 0:
                    print('SGD epoch = %i, grad_norm = %f' %(i, np.linalg.norm(grad)))

    def predict(self, test_set, normalize=False):
        y_pred = []
        for i in np.arange(test_set.shape[1]):
            self.forward_propagate(input_data=test_set[:,i])
            y_hat = self.node_states[-1].copy()
            if normalize:
                y_hat /= np.sum(y_hat)
            y_pred.append(y_hat)
            # print('!! y_hat', y_hat)
        return y_pred


### Helper functions

def loss_function(y, y_hat, type='cross-entropy'):
    """
    y_hat = column array of predictive PMF
    y = column array of one-hot encoding of true class label
    """
    if type == 'cross_entropy':
        return cross-entropy(y=y, y_hat=y_hat)
    elif type == 'square':
        return (1/2) * (y_hat - y).T @ (y_hat - y)


def delta_loss_function(y, y_hat, type='cross-entropy'):
    """
    y_hat = column array of predictive PMF
    y = column array of one-hot encoding of true class label
    """
    # return delta_cross_entropy(y=y, y_hat=y_hat/np.sum(y_hat))

    if type == 'cross-entropy':
        return delta_cross_entropy(y=y, y_hat=y_hat)
    elif type == 'square':
        return y_hat - y
    elif type == 'softmax-cross-entropy':
        return softmax(y_hat) - y


def activation(x, type='sigmoid'):
    if type == 'sigmoid':
        return 1/(1+np.exp(-x))
    elif type == 'ReLU':
        return np.maximum(0,x)
    elif type == 'tanh':
        return (np.exp(2*x)-1)/(np.exp(2*x)+1)

def delta_activation(x, type='sigmoid'):
    # derivate of activation function
    if type == 'sigmoid':
        return sigmoid(x)*(1-sigmoid(x))
    elif type == 'ReLU':
        return int((x>0))
    elif type == 'tanh':
        return (2/(np.exp(x)+np.exp(-x)))**2

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ReLU(x):
    return np.maximum(0,x)

def tanh(x):
    return (np.exp(2*x)-1)/(np.exp(2*x)+1)

def ssoftmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

def cross_entropy(y, y_hat):
    """
    y_hat = column array of predictive PMF
    y = column array of one-hot encoding of true class label
    """
    z = (y.T @ np.log(y_hat))[0][0]
    return (y.T @ np.log(y_hat))[0][0]

def delta_cross_entropy(y, y_hat):
    """
    y_hat = column array of predictive PMF
    y = column array of one-hot encoding of true class label
    """
    y_hat /= np.max(y_hat)
    z = y.copy()
    for i in np.arange(y.shape[0]):
        a = y.argmax(axis=0)[0]
        z[i,0] = -1/y_hat[a, 0]
    return z

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

def softmax(a):
    """
    given an array a = [a_1, .. a_k], compute the softmax distribution p = [p_1, .. , p_k] where p_i \propto exp(a_i)
    """
    a1 = a - np.max(a)
    p = np.exp(a1)
    if type(a) is list:
        p = p/np.sum(p)
    else:
        row_sum = np.sum(p, axis=1)
        p = p/row_sum[:, np.newaxis]
    return p
