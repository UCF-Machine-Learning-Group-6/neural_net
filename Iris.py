import tensorflow as tf
import numpy as np
import pandas as pd

#IMPORTS for Graph:
#import seaborn as sns
#import matplotlib

data=pd.read_csv('C:/iris.data', names=['S_L','S_W','P_L','P_W','Type'])

data["Type"].value_counts()

#map data into arrays
s=np.asarray([1,0,0])
ve=np.asarray([0,1,0])
vi=np.asarray([0,0,1])
data['Type'] = data['Type'].map({'Iris-setosa': s, 'Iris-versicolor': ve,'Iris-virginica':vi})

#shuffle the data
data=data.iloc[np.random.permutation(len(data))]

data=data.reset_index(drop=True)

#training data
x_input=data.loc[0:120,['S_L','S_W','P_L','P_W']]
temp=data['Type']
y_input=temp[0:121]
#test data
x_test=data.loc[121:149,['S_L','S_W','P_L','P_W']]
y_test=temp[121:150]

#placeholders and variables. input has 4 features and output has 3 classes
x=tf.placeholder(tf.float32,shape=[None,4])
y_=tf.placeholder(tf.float32,shape=[None, 3])
#weight and bias
W=tf.Variable(tf.zeros([4,3]))
b=tf.Variable(tf.zeros([3]))
 
#softmax function for multiclass classification
y = tf.nn.softmax(tf.matmul(x, W) + b)

#loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#optimiser -
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
#calculating accuracy of our model 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#session parameters
sess = tf.InteractiveSession()
#initialising variables
init = tf.global_variables_initializer()
sess.run(init)
#number of interations
epoch=5000

for step in range(epoch):
   _, c=sess.run([train_step,cross_entropy], feed_dict={x: x_input, y_:[t for t in y_input.as_matrix()]})
   if step%500==0 :
       print (c)
       
print('Accuracy =',sess.run(accuracy,feed_dict={x: x_test, y_:[t for t in y_test.as_matrix()]}))      

#CODE FOR GRAPH:

#matplotlib.use('Agg')
#matplotlib.style.use('ggplot')
#sns.set()
#df = sns.load_dataset('iris')
#sns_plot = sns.pairplot(df, hue='species', size=2.5)
#fig = sns_plot.savefif("output.png")
#sns.plt.show()
