import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

num_input = 3
num_output = 3
num_hidden1 = 16
num_hidden2 = 16
k_folds = 10
N_EPOCHS = 1000

CROSS_VALIDATION_ACTIVE = True

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

X = tf.placeholder(tf.float32, shape=[None, num_input], name='X')
y = tf.placeholder(tf.float32, shape=[None, num_output], name='y')

#y_dumb = tf.placeholder(tf.float32, shape=[None, num_output], name='y_dumb')
data_csv = pd.read_csv('./data/data.csv')
dataX = data_csv[['in1','in2','in3']]
datay = data_csv[['out1','out2','out3']]
data = data_csv[['in1','in2','in3','out1','out2','out3']]
X_in = []
y_in = []

def shuffle_data():
    global data,dataX,datay,X_in,y_in
    data = data.sample(frac=1).reset_index(drop=True)
    dataX = data[['in1','in2','in3']]
    datay = data[['out1','out2','out3']]

    X_in_df = dataX.values
    y_in_df = datay.values
    X_in = X_in_df.tolist()
    y_in = y_in_df.tolist()

shuffle_data()

FOLDS_SIZE = len(X_in) // k_folds

rmse_a = []
pred_list = []

def cost_func(layer, my_y_true):
    ret = tf.sqrt(tf.reduce_mean(tf.square(layer - my_y_true)))
    return ret


# Neural Network Structure
inputs = X

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

cost = cost_func(output_layer,y)
#session.run(output_layer)
#feed_dict_train = {x: X_train,y_true:y_train}

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
session.run(tf.global_variables_initializer())

def train():
    global rmse_a

    feed_dict_train = {X: X_train, y:y_train}

        
    _, cost_error = session.run([optimizer,cost], feed_dict=feed_dict_train)

    rmse_a += [cost_error]

def predict():
    global pred_list
    feed_dict_test = {X: X_test, y:y_test}
    pred_cost = session.run(cost,feed_dict=feed_dict_test)
    pred_list += [pred_cost]
    print ("Test Error : " , pred_cost )



for epoch in range(N_EPOCHS):
    print ("EPOCH :", epoch+1)
    
    cur_st = 0
    if CROSS_VALIDATION_ACTIVE :
        shuffle_data()
        while True:
            VALIDATION_L = cur_st
            VALIDATION_R = min(cur_st + FOLDS_SIZE,len(X_in))
            #print (VALIDATION_L, VALIDATION_R)
            #print (type(X_in))
            X_train_l = X_in[:VALIDATION_L] + X_in[VALIDATION_R:]
            y_train_l = y_in[:VALIDATION_L] + y_in[VALIDATION_R:]
            X_test_l = X_in[VALIDATION_L:VALIDATION_R]
            y_test_l = y_in[VALIDATION_L:VALIDATION_R]
            #print ("-->",len(X_train),len(X_test) )

            X_train = np.array(X_train_l)
            y_train = np.array(y_train_l)
            X_test = np.array(X_test_l)
            y_test = np.array(y_test_l)

            train()
            predict()

            cur_st += FOLDS_SIZE
            if cur_st >= len(X_in):
                break
    else :
        VALIDATION_NUM = (int) (0.2*len(X_in))
        print (VALIDATION_NUM)
        X_train = X_in[:VALIDATION_NUM]
        y_train = y_in[:VALIDATION_NUM]
        X_test = X_in[VALIDATION_NUM:]
        y_test = y_in[VALIDATION_NUM:]
        train()
        predict()

print (rmse_a[-1])
print (pred_list[-1])
plt.plot(list(enumerate(range(len(rmse_a)))),rmse_a)
#plt.plot(list(enumerate(range(len(pred_list)))),pred_list)
plt.show()
'''
def cross_val(k, _X, _y):
    print (_X)
    try:
        if _X.shape[0] != _y.shape[0]:
            raise ValueError('The size of input and output is not the same !',(_X.shape[0],_y.shape[0]))
    except(ValueError):
        exit('Please check the input and output')      
    return _X, _y  

def test_my_cross_val():
    print ('oraoraoraora')
    #cross_val(10, X, y)
#print (dataX.values)
#cross_val(10, X, y)

feed_dict = {X : dataX.values, y : datay.values}

session.run(tf.global_variables_initializer())

session.run(test_my_cross_val, feed_dict=feed_dict)
'''
'''
feed_dict = {X : X_in, y : y_in}
session.run(tf.global_variables_initializer(), feed_dict=feed_dict)
'''