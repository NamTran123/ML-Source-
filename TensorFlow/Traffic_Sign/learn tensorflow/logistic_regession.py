import  tensorflow  as tf  

#import dataset  MNIST

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/",  one_hot=True)

#Parameters
learning_rate =  0.01
training_epochs  = 50 
batch_size  =  100  
display_step =  1  

#tf Graph  Input 
x  =  tf.placeholder(tf.float32 , [None , 784]) # 28 *28
y  =  tf.placeholder(tf.float32 ,[None,10]) # 10 class 

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#Construct  model  
predic  =  tf.nn.softmax(tf.matmul(x , W) + b )

#Minimize  error  using cross entropy  
cost =  tf.reduce_mean(-tf.reduce_sum(y*tf.log(predic),reduction_indices =1))

#Gradient  Descent  

optimizer  =  tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Imitialize  the variable  
init  = tf.global_variables_initializer()

#Start training  
with tf.Session() as sess:
    sess.run(init)

    #Training cycle 
    for  epoch  in range(training_epochs) :
        avg_cost  =  0
        
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            #Fix training using patch  data  
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer, cost], feed_dict={x: batch_xs,y: batch_ys})
            #Computer  everage  loss  
            avg_cost  += c / total_batch 
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print ("Epoch:",(epoch+1),"cost=",format(avg_cost))
    
    print ("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(predic, 1), tf.argmax(y, 1))
    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))