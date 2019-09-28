###############################################################################
###############################################################################
#
#   utilities.py
#
#   Helper function to build neural network from scratch
#   
#   This python files contains all the helper function necessary to build a fully
#   connected neural network from scratch. A jupyter notebook name main.ipynb will
#   use function definitions written in this file to run the neural netwok.
#   
#
#   by: Pankaj Singh (183079036)
#	    EE, IIT Bombay
#
#   Date: 22nd September, 2019
#
#
#
#
################################################################################
################################################################################



import numpy as np  # only thing we need :)



#############################  initialization  ##############################
def initialization(layers):
	""" This function just initializes weight and biases of the network for
	    give architecture

	 args    :::   layers  -   architecture of FCN
	 returns :::   weights -   dictionary containing weight matrix for all layer intialized by some random number
	 			   biasee  -   dictionary containing bias vectors for all layers initilized to zero
	"""

	weights = {}
	biases = {}

	for i in range(len(layers)-1):

		weights["W" + str(i + 1)] = np.random.randn(layers[i+1], layers[i]) * 0.1
		biases["b" + str(i + 1)] = np.zeros((layers[i+1], 1))

	return weights, biases

#############################################################################



#############################################   forward propagation  ##########################################
# following function will be used during
# forward pass.


def sigmoid(Z):
	"""  This function implements sigmoid function in forward pass
	"""
	A = 1 / ( 1 + np.exp(-Z))

	return A


def relu(Z):
	""" This function implements relu function in forward pass
	"""
	A = np.maximum(0, Z)

	return A



def softmax(Z):
    """
        This function implement softmax function in forward pass

	args    ::::   Z      - (matrix having dimension C * m - where C is no. of classes and m is no of training examples)
	returns ::::   soft_Z - (matrix having same dimension as Z where one column will have score of each class for that example)
    """
    shiftZ = Z - np.max(Z, axis = 0, keepdims = True)          # to avoid exp() function from exploding we normalize the values
    exp_Z = np.exp(shiftZ)									   # exponential of normalized values
    soft_Z = exp_Z / np.sum(exp_Z, axis = 0 , keepdims = True) # calculating softmax scores
  
    return soft_Z



def forward(Aprev, W, b, activation):
	""" this function calculates Z = W*X + b  and passes this Z through sigmoid or relu
		depending upon the value of the activation given('sigmoid', or 'relu')

	args    ::::  Aprev      - Out of previous layer
	              W          - Weigths of current layer
	              b          - Biases of current layer
	              activation -  "sigmoid" or "relu"  (to be used after Z = W*X + b)
	returns ::::  A     -   output of the current layer
 				  cache -   saves some items which will be used in back prop
	"""

	Z = np.dot(W, Aprev) + b      # Linear forward pas

	# supplying the appropriate activation function
	if activation == 'relu':
		A = relu(Z)
	elif activation == 'sigmoid':
		A = sigmoid(Z)
	elif activation == 'softmax':
		A = softmax(Z)
	else:
		raise No_such_activation_fucntion_present


	cache = (Z, Aprev, W, b)      # things we may need during back prop

	return A, cache

#############################################################################################################


###############################################  backward propagation  ######################################
# following function will be need during
# backward propogation step.

def backward_sigmoid(dA, cache):
	""" This funcion calculate gradient flow from sigmoid function. for given input 'Z' output of sigmoid functin
	    was 'A' and their respective gradients are 'dZ' and 'dA'

	args    ::::   dA    - gradient flowing into sigmoid function
	               cache - We need 'Z' from this cache(saved during forward pass)
	returns ::::   dZ    - gradient flowing out of sigmoid function.
	"""

	Z , _, __, ___ = cache                   # we need only Z to calculate dZ from dA
	                                         # rest of values in cache are useless to us
	dZ = dA * sigmoid(Z) * (1-sigmoid(Z))    # simply derived from chain rule

	return dZ




def backward_relu(dA, cache):
	""" This funcion calculate gradient flow from relu function. for given input 'Z' output of relu function
	    was 'A' and their respective gradients are 'dZ' and 'dA'

	args    ::::   dA    - gradient flowing into relu function
	               cache - We need 'Z' from this cache(saved during forward pass)
	returns ::::   dZ    - gradient flowing out of relu function.
	"""

	Z, _, __, ___ = cache

	dZ = dA * ( Z > 0).astype(float)   # gradient is '1' if (Z>0) else 0

	return dZ

def backward_softmax(dA, cache):
	""" This is jus a proxy function. Actually we have considered the gradient of final output directly with
	    wrt input of softmax which makes gradient calculation easier. This was cosidered in gradient
	    initialization itself, But our code was structured to pass the through a  activation backward pass
	    so just taking input and validating it's shape and sending it back.
	"""
	dZ = dA

	return dZ


def backward(dA, cache, activation):
	"""  This function calculates gradient flow through a layer ie. Z = activation_function(WX + b)

	args   ::::     dA          - gradients flowing into the layer
	       ::::     cache       - some items saved during the forwar pass
	                actiavation - activation function used in the given layer
	return ::::     dApre       - gradients flowing out of the layer(will be used by layer before for back prop)
					dW          - gradients of weights
					db          - gradients of biases    
	"""

	Z, Aprev, W, b = cache                         # getting all the items from cache
	m = Z.shape[1]                                 # no of examples

	# supplying the proper activation values and its gradient flow
	if activation == 'sigmoid':
		dZ = backward_sigmoid(dA, cache)
	elif activation == 'relu':
		dZ = backward_relu(dA, cache)
	elif activation == 'softmax':
		dZ = backward_softmax(dA, cache)
	else:
		raise No_such_activation_fucntion_present

	## these three results can be easily obtained from the  vectorized equation ::  Z = np.dot(W Aprev) + b
	## we have  dZ and we derive dW , db and dAprev from this
	dW = np.dot(dZ, Aprev.T) / m
	db = np.sum(dZ, axis = 1, keepdims = True) / m
	dApre = np.dot(W.T, dZ)
	## refer to A. NG notes if in doubt


	return dApre , dW, db



