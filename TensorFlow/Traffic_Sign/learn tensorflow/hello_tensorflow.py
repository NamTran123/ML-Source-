import  tensorflow  as  tf  

#op is added  as a node  to the  default  graph 
hello  =  tf.constant('Hello , Tensorflow')

#Start  tf  session  
sess  = tf.Session()

#run  op 

print( sess.run(hello))

