import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MY_BOUNDED_SET = 40000
num_input = 3
num_output = 3
num_hidden1 = 16
num_hidden2 = 16
N_EPOCHS = 10
epoch = 0
train_rmse_list = []
test_rmse_list = []

session = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, num_input])

y_true = tf.placeholder(tf.float32, shape=[None, num_output], name='y_true')
y_pred = tf.placeholder(tf.float32, shape=[None, num_output])

data_csv = pd.read_csv('./data/data.csv')
X = data_csv[['in1','in2','in3']]
y = data_csv[['out1','out2','out3']]

X_train_pd = X[:MY_BOUNDED_SET]
X_test_pd = X[MY_BOUNDED_SET:]
y_train_pd = y[:MY_BOUNDED_SET]
y_test_pd = y[MY_BOUNDED_SET:]

X_train = X_train_pd.values
X_train = X_train.astype(np.float32)

y_train = y_train_pd.values
y_train = y_train.astype(np.float32)

X_test = X_test_pd.values
X_test = X_test.astype(np.float32)

y_test = y_test_pd.values
y_test = y_test.astype(np.float32)

rmse_a = []
pred_list = []

def cost_func(layer, my_y_true):
    ret = tf.sqrt(tf.reduce_mean(tf.square(layer - my_y_true)))
    return ret


# Neural Network Structure
inputs = X_train

# Input Layer
input_layer = tf.Variable(inputs)

# Hidden Layer #1
h1 = tf.layers.dense(inputs=input_layer, 
                     units=num_hidden1,
                     use_bias=True,
                     activation=tf.nn.relu)
   
# Hidden Layer #2
h2 = tf.layers.dense(inputs=h1, 
                     units=num_hidden2,
                     use_bias=True,
                     activation=tf.nn.relu)
    
# Output Layer
output_layer = tf.layers.dense(inputs=h2, 
                     units=num_output,
                     use_bias=True,
                     activation=None)

cost = cost_func(output_layer,y_true)
#session.run(output_layer)
feed_dict_train = {x: X_train,y_true:y_train}

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
session.run(tf.global_variables_initializer())

def train(N_EPOCHS):
    global rmse_a
    global pred_list
    for i in range(epoch, epoch + N_EPOCHS):
        feed_dict_train = {x: X_train,
                      y_true:y_train}

        
        _, cost_error, pred_layer = session.run([optimizer,cost, output_layer], feed_dict=feed_dict_train)
        #cost_error = session.run(cost,feed_dict=feed_dict_train)
        #print ("[" +str(i+1) +"]")
        print ("Train Accuracy : " , cost_error)
        
        rmse_a += [cost_error]
        pred_list += [pred_layer]

def predict():
    session.run(tf.global_variables_initializer())
    feed_dict_test = {x: X_test,
                    y_true:y_test}
    print ("Test Accuracy : " , session.run(cost,feed_dict=feed_dict_test))

train(N_EPOCHS)
#predict()

plt.plot(list(enumerate(range(len(rmse_a)))),rmse_a)
print (rmse_a[-1])
print (pred_list[-1])
#print (len(pred_list),len(pred_list[0])) # EPOCH, BOUND_SET_NUM