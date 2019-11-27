import numpy as np                      # scientific computing lib
import matplotlib.pyplot as plt         # to interact with dataset that is stored on an H5 file
import h5py                             # library to plot graphs in Python
import scipy
import cv2
from PIL import Image                   # Pillow = Python image library
from scipy import ndimage, misc               # Pillow and scipy are used to test our model with my own picture at the end
from lr_utils import load_dataset

# %matplotlib inline

# Loading the data(cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of picture
# index = 1
# plt.imshow(train_set_x_orig[index])
# print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "'picture.")
# plt.show()

# Remember that `train_set_x_orig` is a numpy-array of shape (m_train, num_px, num_px, 3).
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print("Number of training examples: m_train = " + str(m_train))
print("Number of test examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))

 # Reshape the training and test data sets so that images of size (num_px, num_px, 3) are flattened into
 # single vectors of shape (num_px  ∗∗  num_px  ∗∗  3, 1).
 # A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b ∗∗ c ∗∗ d, a) is to use:
 # X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y.shape))
print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

# To represent color images, the red, green and blue channels (RGB) must be specified for each pixel,
# and so the pixel value is actually a vector of three numbers ranging from 0 to 255.
#
# One common preprocessing step in machine learning is to center and standardize your dataset, meaning that you substract
# the mean of the whole numpy array from each example, and then divide  each example by the standard deviation of the whole numpy array.
# But for picture datasets, it is simpler and more convenient and works almost as well to just divide every row of the dataset
# by 255 (the maximum value of a pixel channel).

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- scalar or numpy array of any standardize

    Return:
    s -- sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    return s

# print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))         # sigmoid test

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape(dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    w = np.zeros((dim, 1), dtype = float)
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b

# dim = 2                                                       # initialize_with_zeros test
# w, b = initialize_with_zeros(dim)
# print("w = " + str(w))
# print("b = " + str(b))

""" Forward and Bacward propagation """

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Returns:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss witn respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]

    # forward propagation -- from x to cost
    A = sigmoid(np.dot(w.T, X) + b)                                                # compute activation
    cost = -np.sum(np.dot(Y, np.log(A).T) + np.dot(1-Y, np.log(1-A).T)) / m         # compute cost

    # bacward propagation -- to find gradients
    dw = np.dot(X, (A-Y).T) / m
    db = np.sum(A-Y) / m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


# w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
# grads, cost = propagate(w, b, X, Y)
# print("dw = " + str(grads["dw"]))
# print("db = " + str(grads["db"]))
# print("cost = " + str(cost))

""" We have initialized your parameters.
    We are also able to compute a cost function and its gradient.
    Now, you want to update the parameters using gradient descent. """

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector(containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing gradients of the weights and bias with respect to cost function
    costs -- list of all the costs computed during optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1-) Calculate the cost and the gradient for the current parameters. Use propagate().
        2-) Update the parameters using gradient descent rule for w and b.
    """

    costs = []

    for i in range(num_iterations):
        # cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)

        # retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule   # parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

# params, grads, costs = optimize(w, b, X, Y, num_iterations = 100, learning_rate = 0.009, print_cost = False)
# print("w = " + str(params["w"]))
# print("b = " + str(params["b"]))
# print("dw = " + str(grads["dw"]))
# print("db = " + str(grads["db"]))

""" The previous function will output the learned w and b. We are able to use w and b to predict the labels for a dataset X.
    Implement the predict() function. There are two steps to computing predictions:
    Calculate  Ŷ =A=σ(wTX+b)Y^=A=σ(wTX+b)
    Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5), stores the predictions in a vector Y_prediction.
    If you wish, you can use an if/else statement in a for loop (though there is also a way to vectorize this). """

def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_predictions -- a numpy array (vector) containing all predictions2 (either 1 or 0) for the examples in X
    """

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0,i] >= 0.5:
            Y_prediction[0,i] = 1

    assert(Y_prediction.shape == (1, m))

    return Y_prediction

# w = np.array([[0.1124579], [0.23106775]])
# b = -0.3
# X = np.array([[1.,-1.1, -3.2], [1.2, 2., 0.1]])
# print("predictions = " + str(predict(w, b, X)))

""" Implement the model function. Use the following notation:

    - Y_prediction_test for your predictions on the test set
    - Y_prediction_train for your predictions on the train set
    - w, costs, grads for the outputs of optimize() """

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the functions we've implemented previously

    Arguments:
    X_train -- training set represented bu a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in update rule of optimize()
    print_cost -- set to true to print the cost every 100 num_iterations

    Returns:
    d -- dictionary containing information about the model
    """

    # initialize parameters with zetos
    w, b = initialize_with_zeros(X_train.shape[0])

    # gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = print_cost)

    # retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # predict train/test set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

"""
    Comment: Training accuracy is close to 100%. This is a good sanity check: your model is working and has high enough capacity
    to fit the training data. Test accuracy is 68%. It is actually not bad for this simple model, given the small dataset we used
    and that logistic regression is a linear classifier. But no worries, you'll build an even better classifier next week!

    Also, you see that the model is clearly overfitting the training data. Later in this specialization
    you will learn how to reduce overfitting, for example by using regularization. """

# # Example of a picture that was wrongly classified.
index = 30
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
# print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a '" + classes[d["Y_prediction_test"][0,index]].decode("utf-8") +  "' picture.")
print ("y = " + str(test_set_y[0,index]))
plt.show()


# Let's also plot the cost function and the gradients.
# Plot learning curve (with costs)

costs = np.squeeze(d["costs"])
plt.plot(costs)
plt.ylabel("cost")
plt.xlabel("iterations (per hundreds)")
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

# Interpretation: You can see the cost decreasing. It shows that the parameters are being learned.
# However, you see that you could train the model even more on the training set.
# Try to increase the number of iterations in the cell above and rerun the cells.
# You might see that the training set accuracy goes up, but the test set accuracy goes down. This is called overfitting.


# Let's analyze it further, and examine possible choices for the learning rate.
# Reminder: In order for Gradient Descent to work you must choose the learning rate wisely.
# The learning rate  αα  determines how rapidly we update the parameters.
# If the learning rate is too large we may "overshoot" the optimal value.
# Similarly, if it is too small we will need too many iterations to converge to the best values.
# That's why it is crucial to use a well-tuned learning rate.
# Let's compare the learning curve of our model with several choices of learning rates.

learning_rates = [0.01, 0.001, 0.0001]
models = {}

for i in learning_rates:
    print("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print("\n" + "------------------------------------------------------" + "\n")

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label = str(models[str(i)]["learning_rate"]))

plt.ylabel("cost")
plt.xlabel("iterations (hundreds)")

legend = plt.legend(loc = "upper center", shadow = True)
frame = legend.get_frame()
frame.set_facecolor("0.90")
plt.show()

# Interpretation:
#
# Different learning rates give different costs and thus different predictions results.
# If the learning rate is too large (0.01), the cost may oscillate up and down.
# It may even diverge (though in this example, using 0.01 still eventually ends up at a good value for the cost).
# A lower cost doesn't mean a better model. You have to check if there is possibly overfitting.
# It happens when the training accuracy is a lot higher than the test accuracy.
# In deep learning, we usually recommend that you: Choose the learning rate that better minimizes the cost function.
# If your model overfits, use other techniques to reduce overfitting.

""" Test with your own image """
# my_image = "cropped headshot.jpg"
my_image = "cat-test.jpg"
# Preprocess the image to fit our algorithm.
fname = "images/" + my_image
image = np.array(plt.imread(fname))
my_image = cv2.resize(image, (64, 64))
plt.imshow(my_image)
plt.show()
my_image = my_image.reshape(1, num_px * num_px * 3).T
my_image = my_image / 255
my_predicted_image = predict(d["w"], d["b"], my_image)
print("y = " + str(np.squeeze(my_predicted_image))  + ", your algorithm predicts a '" + classes[int(np.squeeze(my_predicted_image))].decode("utf-8") +  "\" picture.")
