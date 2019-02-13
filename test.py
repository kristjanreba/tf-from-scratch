'''
import tensorflow as tf

with tf.Session() as sess:
    a = tf.constant(3, name='a')
    b = tf.constant(5, name='b')
    prod = tf.multiply(a, b, name='Multiply')
    sum = tf.add(a, b, name='Add')
    res = tf.divide(prod, sum, name='divide')

    out = sess.run(res)
    print(out)
'''

import tf_api as tf

# create default graph
tf.Graph().as_default()

# construct computational graph by creating some nodes
a = tf.Constant(15)
b = tf.Constant(5)
prod = tf.multiply(a, b)
sum = tf.add(a, b)
res = tf.divide(prod, sum)

# create a session object
session = tf.Session()

# run computational graph to compute the output for 'res'
out = session.run(res)
print(out)
