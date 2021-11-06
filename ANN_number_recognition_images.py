########################################################################################################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                                     ANN for number recognition from Images
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
########################################################################################################################################################

# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others

import random

# tells matplotlib to embed plots within the notebook
%matplotlib inline

###################################################################################################################

def number_pred(x):
    
    a1 = np.transpose(tem)
    z2 = np.dot(Theta1,a1)
    a2 = sigmoid(z2)
    a2 = np.r_[1,a2]
    z3 = np.dot(Theta2,a2)
    hx = sigmoid(z3)
    
    return np.where(hx == max(hx))

###################################################################################################################

def displayData(X, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data stored in X in a nice grid.
    """
    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = pyplot.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        # Display Image
        h = ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                      cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')

###################################################################################################################

def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_=0.0):
    
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    # Setup some useful variables
    m = y.size
         
    # You need to return the following variables correctly 
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # ====================== YOUR CODE HERE ======================

    tX = np.c_[np.ones((m,1)),X]
    y_mat = np.transpose([range(0,10)]) == y
    
    #
    #====================== FORWARD PROPAGATION ========================
    #
    
    a1 = np.transpose(tX)
    z2 = np.dot(Theta1,a1)
    a2 = sigmoid(z2)
    a2 = np.r_[ np.ones((1,m)) , a2]
    z3 = np.dot(Theta2,a2)
    hx = sigmoid(z3)
    
    #
    #====================== COST FUNCTION ========================
    #
    
    #J = np.sum((1/m) * (-1*y_mat * np.log(hx) - (1-y_mat) * np.log(1-hx) ) ) + (lambda_/(2*m)) * (np.sum(pow(Theta1[:,1:] , 2)) + np.sum(pow(Theta2[:,1:] , 2)) )
    J = (-1/m) * np.sum(y_mat * np.log(hx) + (1-y_mat) * np.log(1-hx)) + (lambda_/(2*m)) * (np.sum(pow(Theta1[:,1:] , 2)) + np.sum(pow(Theta2[:,1:] , 2)) )

    
    #
    #====================== BACKWARD PROPAGATION ========================
    #
    
    delta3 = hx - y_mat
    z2 = np.r_[np.ones((1,m)),z2]
    delta2 = np.dot(np.transpose(Theta2),delta3) * sigmoid(z2)*(1-sigmoid(z2))
    delta2 = delta2[1:hidden_layer_size+1,:]
    Theta1_g = np.dot(delta2 , np.transpose(a1))
    Theta2_g = np.dot(delta3 , np.transpose(a2))
    
    Theta1_grad = (1 / m) * Theta1_g
    Theta2_grad = (1 / m) * Theta2_g
 
    Theta1_grad[:,1:] = (1) * ( Theta1_grad[:,1:] + (lambda_/m) * Theta1[:,1:] )
    Theta2_grad[:,1:] = (1) * ( Theta2_grad[:,1:] + (lambda_/m) * Theta2[:,1:] )
    
    # ================================================================
    # Unroll gradients
    # grad = np.concatenate([Theta1_grad.ravel(order=order), Theta2_grad.ravel(order=order)])
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

    return J, grad

###################################################################################################################

def sigmoidGradient(z):
    
    g = np.zeros(z.shape)

    g = sigmoid(z) * (1 - sigmoid(z))

    return g

###################################################################################################################

def randInitializeWeights(L_in, L_out, epsilon_init=0.12):
  
    W = np.zeros((L_out, 1 + L_in))
    epsilon_init = 0.12
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

    return W

###################################################################################################################

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

###################################################################################################################

def predict(Theta1, Theta2, X):
    """
    Predict the label of an input given a trained neural network
    Outputs the predicted label of X given the trained weights of a neural
    network(Theta1, Theta2)
    """
    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly
    p = np.zeros(m)
    h1 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), X], axis=1), Theta1.T))
    h2 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), Theta2.T))
    p = np.argmax(h2, axis=1)
    return p

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                                                   IMPORT TRAINING DATA [X y]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#  training data stored in arrays X, y
data = loadmat(os.path.join('/content/drive/MyDrive/Colab Notebooks/ML work/ex4data1.mat'))
X, y = data['X'], data['y'].ravel()

# MATLAB where there is no index 0
y[y == 10] = 0

# Number of training examples
m = y.size

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                                            RAANDOM INITIALIZATION OF NEURAL NETWORK PARAMETERS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


print('Initializing Neural Network Parameters ...')

# Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                                             GENERATE NEURAL NETWORK PARAMETERS USING SCIPY
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Set number of iterations
options= {'maxiter': 500}

#  Regularization parameter lambda
lambda_ = 1

# Create "short hand" for the cost function to be minimized
costFunction = lambda p: nnCostFunction(p, input_layer_size,
                                        hidden_layer_size,
                                        num_labels, X, y, lambda_)

# nncostFunction is a function that takes in only one argument
# (the neural network parameters)
res = optimize.minimize(costFunction,
                        initial_nn_params,
                        jac=True,
                        method='TNC',
                        options=options)

# get the solution of the optimization
nn_params = res.x
        
# Obtain Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))

Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                    (num_labels, (hidden_layer_size + 1)))

print('Done')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                                             IMAGE INPUT AND FINAL PREDICTION
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from PIL import Image

image = Image.open('/content/drive/MyDrive/Colab Notebooks/ML work/Number recognition samples/NN_Trial_7.png')      # import image to read
print(image.size)
#image.rotate(-90)
new_image = image.resize((20, 20))
print(new_image.size)
greyscale_image = new_image.convert('L')
img_np = np.asarray(greyscale_image)
img_mat = img_np.ravel()
img_mat.shape
greyscale_image

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

p = predict(Theta1,Theta2,X)
print('Training Set Accuracy: %f' % (np.mean(p == y) * 100))

n = random.randint(0, 4999)
#a = (X[n,:])
a = img_mat
tem = (np.r_[1,a])   
    
value, = number_pred(tem)
print(value)
displayData(a)
