import math

import numpy as np
from datasets import generate_nd_dataset, kGaussian, train_test_split_
from random_control import generator
from losses import negative_log_likelihood, nll_derivative



class Layer:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons

    def softmax(self, inputs):
        # TODO: YOUR CODE HERE # (Part 2)
        #apply softmax to the final output
        #normalize vector of real numbers into probability distribution
        #first we have vector of all elements as an exponent of e
        exps = np.exp(inputs)
        sum = np.sum(exps)
        ret = np.ndarray(i / sum for i in exps)
        #CHECK THIS to make sure the idea works
        print(ret)
        #check again -- do all the probabilities add up to 1?
        print(np.sum(ret))
        return np.ndarray(ret)
        # END OF YOUR CODE #

    def sigmoid(self, inputs):
        # TODO: YOUR CODE HERE # (Part 2)
        #the sigmoid function is 1/(1+e^-x)
        return 1 / (1 + np.exp(-input))
        # END OF YOUR CODE #

    def sigmoid_derivative(self, Z):
        # TODO: YOUR CODE HERE # (Part 2)
        
        return -np.exp(Z) / (1+np.exp(Z))**2
        # END OF YOUR CODE #
    
    def tanH(self, inputs):
        return np.tanh(inputs)

    def tanH_derivative(self, Z):
        # TODO: YOUR CODE HERE # (Part 5)
        return 1 - np.tanh(Z)**2
        # END OF YOUR CODE #

    def relu(self, inputs):
        # TODO: YOUR CODE HERE # (Part 5)
        return None
        # END OF YOUR CODE #

    def relu_derivative(self, Z):
        # TODO: YOUR CODE HERE # (Part 5)
        return None
        # END OF YOUR CODE #

    # TODO: YOUR CODE HERE #
    def apply_chain_rule_activation_derivative(self, z, activation_derivative):
        # rename the variable q appropriately -- what should this be?
        #q should be z
        # then, apply the chain rule and return the result
        return None

    # END OF YOUR CODE #

    def forward(self, inputs, weights, bias, activation):
        # TODO: YOUR CODE HERE #
        #arrays/matrices: inputs, weights
        #scalars: bias, activation

        #z_j = sum_i(w_ij * a_i) * b_j
        """
        for output z in layer l, 
        i goes from 1 to m^l (the size of the layer)
        z = the sum of (all weights in the same row to this point 
            * all of the activations to this point) * current bias
        for each 
        """
        
        Z_curr = 0
        for i in range(len(weights)):
            #wij*ai
            Z_curr += weights[i] * activation[i] 
        Z_curr += bias
          # compute Z_curr from weights and bias
        # END OF YOUR CODE #

        if activation == "relu":
            A_curr = self.relu(inputs=Z_curr)
        elif activation == "sigmoid":
            A_curr = self.sigmoid(inputs=Z_curr)
        elif activation == "tanH":
            A_curr = self.tanH(inputs=Z_curr)
        elif activation == "softmax":
            A_curr = self.softmax(inputs=Z_curr)
        else:
            raise ValueError("Activation function not supported: " + activation)

        return A_curr, Z_curr

    def backward(self, dA_curr, W_curr, Z_curr, A_prev, activation):
        """
        :param dA_curr: the partial derivative of the loss with respect to the activation of the preceding layer (l + 1)
        :param W_curr: the weights of the layer (l)
        :param Z_curr: the weighted sum of layer (l)
        :param A_prev: the activation of this layer (l) ... we use prev with respect to dA_curr
        :param activation: a string that specifies the activation function used in the layer
        :return: (dA, dW, dB)
                 dA: the activation of this layer (l) -- needed to continue the backprop
                 dW: the weights -- needed to update the weights
                 db: the bias -- (needed to update the bias)
        """
        # TODO Each of these functions require you to compute all the colored terms in the Part 2 Figure.
        #  We will denote the partial derivative of the loss with respect to each variable as dZ, dW, db, dA
        #  These variable map to the corresponding terms in the figure. Note that these are matrices and not individual
        #  values, you will determine how to vectorize the code yourself. Think carefully about dimensions!
        #  You can use the self.apply_chain_rule_activation_derivative() function, but there are solutions without it.
        #  Computing dZ is not technically needed, but it can be used to help compute the other values.
        if activation == "softmax":
            # TODO: YOUR CODE HERE #
            # We deal with the softmax function for you, so dZ is not needed for this one. dA_curr = dZ for this one.
            dW = None
            db = None
            dA = None
            # END OF YOUR CODE #
        elif activation == "sigmoid":
            # TODO: YOUR CODE HERE #
            activation_derivative = None
            dZ = None
            dW = None
            db = None
            dA = None
            # END OF YOUR CODE #
        elif activation == "tanH":
            # TODO: YOUR CODE HERE #
            activation_derivative = None
            dZ = None
            dW = None
            db = None
            dA = None
            # END OF YOUR CODE #
        elif activation == "relu":
            # TODO: YOUR CODE HERE #
            activation_derivative = None
            dZ = None
            dW = None
            db = None
            dA = None
            # END OF YOUR CODE #
        else:
            raise ValueError("Activation function not supported: " + activation)

        return dA, dW, db


class MLP:
    """
    * `MLP` is a class that represents the multi-layer perceptron with a variable number of hidden layer.
       The constructor initializes the weights and biases for the hidden and output layers.
    * `sigmoid`, `relu`, `tanh`, and `softmax` are activation function used in the MLP.
       They should each map any real value to a value between 0 and 1.
    * `forward` computes the forward pass of the MLP.
       It takes an input X and returns the output of the MLP.
    * `sigmoid_derivative`, `relu_derivative`, `tanH_derivative` are the derivatives of the activation functions.
       They are used in the backpropagation algorithm to compute the gradients.
    *  `mse_loss`, `hinge_loss`, `cross_entropy_loss` are each loss functions.
       The MLP algorithms optimizes to minimize those.
    * `backward` computes the backward pass of the MLP. It takes the input X, the true labels y,
       the predicted labels y_hat, and the learning rate as inputs.
       It computes the gradients and updates the weights and biases of the MLP.
    * `train` trains the MLP on the input X and true labels y. It takes the number of epochs
    """

    def __init__(self, layer_list):
        """
        :param layer_list: a list of numbers that specify the width of the hidden layers.
               The dataset dimensionality (input layer) and output layer (1)
               should not be specified.
        """
        self.layer_list = layer_list
        self.network = []  ## layers
        self.architecture = []  ## mapping input neurons --> output neurons
        self.params = []  ## W, b
        self.memory = []  ## Z, A
        self.gradients = []  ## dW, db
        self.all_gradients = []  ## copy of dW, db of the first layer in each update step
        self.loss = []
        self.accuracy = []

        self.loss_func = None
        self.loss_derivative = None

        # For plotting purposes
        self.prediction_loss = None
        self.prediction_accuracy = None
        self.prediction_error = None

        self.init_from_layer_list(self.layer_list)

    # TODO read and understand the next several functions, you will need to understand them to complete the assignment.
    #  In particular, you will need to understand self.network, self.architecture, self.params, self.memory,
    #  and self.gradients. It may be helpful to write some notes about what each of these variables are and how they
    #  are used.
    def init_from_layer_list(self, layer_list):
        for layer_size in layer_list:
            self.add(Layer(layer_size))

    def add(self, layer):
        self.network.append(layer)

    def _compile(self, data, activation_func="relu"):
        self.architecture = []
        for idx, layer in enumerate(self.network):
            if idx == 0:
                self.architecture.append(
                    {
                        "input_dim": data.shape[1],
                        "output_dim": self.network[idx].num_neurons,
                        "activation": activation_func,
                    }
                )
            elif idx > 0 and idx < len(self.network) - 1:
                self.architecture.append(
                    {
                        "input_dim": self.network[idx - 1].num_neurons,
                        "output_dim": self.network[idx].num_neurons,
                        "activation": activation_func,
                    }
                )
            else:
                self.architecture.append(
                    {
                        "input_dim": self.network[idx - 1].num_neurons,
                        "output_dim": self.network[idx].num_neurons,
                        "activation": "softmax",
                    }
                )
        return self

    def _init_weights(self, data, activation_func, seed=None):
        self.params = []
        self._compile(data, activation_func)

        if seed is None:
            for i in range(len(self.architecture)):
                self.params.append(
                    {
                        "W": generator.uniform(
                            low=-1,
                            high=1,
                            size=(
                                self.architecture[i]["input_dim"],
                                self.architecture[i]["output_dim"],
                            ),
                        ),
                        "b": np.zeros((1, self.architecture[i]["output_dim"])),
                    }
                )
        else:
            # For testing purposes
            fixed_generator = np.random.default_rng(seed=seed)
            for i in range(len(self.architecture)):
                self.params.append(
                    {
                        "W": fixed_generator.uniform(
                            low=-1,
                            high=1,
                            size=(
                                self.architecture[i]["input_dim"],
                                self.architecture[i]["output_dim"],
                            ),
                        ),
                        "b": np.zeros((1, self.architecture[i]["output_dim"])),
                    }
                )

        return self

    def forward(self, data):
        A_prev = data
        A_curr = None

        for i in range(len(self.params)):
            # TODO: YOUR CODE HERE #
            #  compute the forward for each layer and store the appropriate values in the memory.
            #  We format our memory_list as a list of dicts, please follow this format.
            #  mem_dict = {'?': ?}; self.memory.append(mem_dict)
            pass
            # END OF YOUR CODE #

        return A_curr

    def backward(self, predicted, actual):
        # compute the gradient on predictions
        dscores = self.loss_derivative(predicted, actual)

        dA_prev = dscores  # This is the derivative of the loss function with respect to the output of the last layer

        # TODO: YOUR CODE HERE #
        #  compute the backward for each layer and store the appropriate values in the gradients.
        #  We format our gradients_list as a list of dicts, please follow this format (same as self.memory).
        #  Also remember to save the gradients in all_gradients
        pass
        # END OF YOUR CODE #

    def _update(self, lr):
        # TODO: YOUR CODE HERE #
        #  update the network parameters using the gradients and the learning rate.
        #  Recall gradients is a list of dicts, and params is a list of dicts, pay attention to the order of the dicts.
        #  Is gradients the same order as params? This might depend on your implementations of forward and backward.
        #  Should we add or subtract the deltas?
        pass
        # END OF YOUR CODE #

    def _calculate_accuracy(self, predicted, actual):
        return np.mean(np.argmax(predicted, axis=1) == actual)

    def _calculate_loss(self, predicted, actual):
        return self.loss_func(predicted, actual)

    def _set_loss_function(self, loss_func_name="negative_log_likelihood"):
        if loss_func_name == "negative_log_likelihood":
            self.loss_func = negative_log_likelihood
            self.loss_derivative = nll_derivative
        else:
            raise Exception("Loss has not been specified. Abort")

    def get_losses(self):
        if len(self.loss) > 0:
            return self.loss
        else:
            return [np.inf]

    def get_accuracy(self):
        if len(self.accuracy) > 0:
            return self.accuracy
        else:
            return [0]

    def train(
        self,
        X_train,
        y_train,
        epochs=1000,
        lr=1e-4,
        batch_size=16,
        activation_func="relu",
        loss_func="negative_log_likelihood",
    ):
        self._set_loss_function(loss_func)  # set loss function
        self.loss = []  # reset every time train is called
        self.accuracy = []  # reset every time train is called

        # cast to int
        y_train = y_train.astype(int)

        # initialize network weights
        self._init_weights(X_train, activation_func)

        # TODO: YOUR CODE HERE #
        # calculate number of batches
        num_datapoints = None
        num_batches = None
        # make sure you handle the case where num_datapoints is not divisible by batch_size, don't lose data!
        # END OF YOUR CODE #

        # NOTE: only reset all_gradients with every call to train
        self.all_gradients = []  # reset every time train is called

        # TODO: shuffle the data and iterate over mini-batches for each epoch.
        #  We are implementing mini-batch gradient descent.
        #  How you batch the data is up to you, but you should remember
        #  shuffling has to happen the same way for both X and y.

        for i in range(int(epochs)):

            batch_loss = 0
            batch_acc = 0

            # TODO: YOUR CODE HERE #
            #  shuffle the data
            pass
            # END OF YOUR CODE #

            for batch_num in range(num_batches - 1):
                # TODO: YOUR CODE HERE #
                X_batch = None
                y_batch = None
                # END OF YOUR CODE #

                # TODO: YOUR CODE HERE #
                #  do any variables need to be reset each pass?
                pass
                # END OF YOUR CODE #

                # TODO: YOUR CODE HERE #
                yhat = None  # TODO compute yhat
                
                acc = None  # TODO compute and update batch acc
                loss = None  # TODO compute and update batch loss
                # END OF YOUR CODE #

                # Stop training if loss is NaN, why might the loss become NaN or inf?
                if np.isnan(loss):
                    acc = 0 if len(self.accuracy) == 0 else self.accuracy[-1]
                    loss = np.inf if len(self.loss) == 0 else self.loss[-1]
                    s = "EPOCH: {}, LR: {}, ACCURACY: {}, LOSS: {}".format(
                        i, lr, acc, loss
                    )
                    print(s)
                    print("Stopping training because loss is NaN")
                    return

                # TODO: YOUR CODE HERE #
                #  update the network
                pass
                # END OF YOUR CODE #

            self.loss.append(batch_loss / num_batches)
            self.accuracy.append(batch_acc / num_batches)

            if i % 20 == 0:
                s = "EPOCH: {}, LR: {}, ACCURACY: {}, LOSS: {}".format(
                    i, lr, self.accuracy[-1], self.loss[-1]
                )
                print(s)

    def predict(self, X, y, loss_func="negative_log_likelihood"):
        y = y.astype(int)
        # TODO: YOUR CODE HERE #
        #  predict the loss and accuracy on a val or test set and print the results. Make sure to gracefully handle
        #  the case where the loss is NaN or inf.
        pass
        # END OF YOUR CODE #

        # for plotting purposes
        # TODO: YOUR CODE HERE #
        self.prediction_loss = None  # TODO loss
        self.prediction_accuracy = None  # TODO accuracy
        self.prediction_error = None  # TODO fill in error: 1 - accuracy
        # END OF YOUR CODE #


if __name__ == "__main__":
    # X, y = sklearn.datasets.load_iris(return_X_y=True)
    N = 200
    M = 200
    dims = 3
    gaus_dataset_points = generate_nd_dataset(N, M, kGaussian, dims).get_dataset()
    X = gaus_dataset_points[:, :-1]
    y = gaus_dataset_points[:, -1].astype(int)
    # model = BinaryMLP([6, 8, 10, 3])
    X_train, X_test, y_train, y_test = train_test_split_(gaus_dataset_points)

    model = MLP([6, 8, 10, 2])
    model.train(X_train, y_train, batch_size=16, epochs=250, lr=1e-2)
    model.predict(X_test, y_test)
