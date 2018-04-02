import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MY_BOUNDED_SET = 45000
num_input = 3
num_output = 3
num_hidden1 = 16
num_hidden2 = 16
N_EPOCHS = 10000
epoch = 0
train_rmse_list = []
test_rmse_list = []


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

x = tf.placeholder(tf.float32, shape=[None, num_input], name='x')

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
inputs = x

# Input Layer
input_layer = inputs
#input_layer = tf.Variable(inputs)

# Hidden Layer #1
h1 = tf.layers.dense(inputs=input_layer, 
                     units=num_hidden1,
                     use_bias=True,
                     activation=tf.nn.elu)
   
# Hidden Layer #2
h2 = tf.layers.dense(inputs=h1, 
                     units=num_hidden2,
                     use_bias=True,
                     activation=tf.nn.elu)
    
# Output Layer
output_layer = tf.layers.dense(inputs=h2, 
                     units=num_output,
                     use_bias=True,
                     activation=None)

cost = cost_func(output_layer,y_true)
#session.run(output_layer)
#feed_dict_train = {x: X_train,y_true:y_train}

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
session.run(tf.global_variables_initializer())

def train(N_EPOCHS):
    global rmse_a

    for i in range(epoch, epoch + N_EPOCHS):
        feed_dict_train = {x: X_train,
                      y_true:y_train}

        
        _, cost_error = session.run([optimizer,cost], feed_dict=feed_dict_train)
        #cost_error = session.run(cost,feed_dict=feed_dict_train)
        #print ("[" +str(i+1) +"]")
        rmse_a += [cost_error]
        if ((i%1000) == 0):
            print ("Train Error : " , cost_error)

def predict():
    global pred_list
    feed_dict_test = {x: X_test,
                    y_true:y_test}
    pred_cost = session.run(cost,feed_dict=feed_dict_test)
    pred_list += [pred_cost]
    print ("Test Error : " , pred_cost )

train(N_EPOCHS)
predict()

print (rmse_a[-1])
print (pred_list[-1])
plt.plot(list(enumerate(range(len(rmse_a)))),rmse_a)
plt.show()
#print (len(pred_list),len(pred_list[0])) # EPOCH, 40000