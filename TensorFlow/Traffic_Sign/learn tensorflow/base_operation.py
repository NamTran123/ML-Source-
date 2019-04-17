# Base Operation  example  using Tensorflow  Library

import tensorflow as tf

# Base  constant operations

a = tf.constant(2)
b = tf.constant(3)

# Launch the  default graph
with tf.Session() as Sess:
    print('a =2  , a=3')
    print('Addition  with constants ', Sess.run(a+b))
    print('Mutiplication with  constants ', Sess.run(a*b))


# tf Graph input
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operation

add = tf.add(a, b)
mul = tf.multiply(a, b)

# Lauch  the default  graph

with tf.Session() as sess:
    print('Addition  with variable', sess.run(add , feed_dict={a: 2, b: 3}))
    print('Mutiplication with  vairiable', sess.run(mul, feed_dict = {a: 2, b: 3}))

#Matrix  Mutiplication  from TensorFlow  official  tutorial 

#Creat  a Constants  of that  produces a  1x2 matrix  . The  op  is  added  as a node  to the default  graph  

matrix  =  tf.constant([[3.,3.]])

#Create another  Constant  that  produces
matrix2  = tf.constant([[2.],[2.]])

#Mutiplication  
product = tf.matmul(matrix , matrix2)

with tf.Session() as ses:
    result =  ses.run(product)
    print(result)ych