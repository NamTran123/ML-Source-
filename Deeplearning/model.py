import  numpy as  np
import  matplotlib.pyplot  as  plt

x = np.array(([0.50], [0.75], [1.00], [1.25], [1.50], [1.75], [1.75], [2.00], [2.25], [2.50], [2.75], [3.00], [3.25], [3.50], [4.00], [4.25], [4.50], [4.75], [5.00], [5.50]),dtype=float )
y = np.array(([0], [0], [0], [0], [0], [0], [1], [0], [1], [0], [1], [0], [1], [0], [1], [1], [1], [1], [1], [1]), dtype=float)


def sigmod(x):
    return (1 / (1 + np.exp(-x)))

def calculate_z(z):
    return  (z*(1-z))

class  Neural_network:
    def  __init__(self ,x ,y):
        self.input  = x
        self.weight1  =  np.random.rand(self.input.shape[1],30)
        self.weight2  = np.random.rand(30,30)
        self.weight3  =  np.random.rand(30,1)
        self.y = y
        self.output  =  np.zeros(y.shape)

    def feedforward (self ):
        a  = np.dot(self.input, self.weight1)
        self.layer1  =  sigmod(a)
        b = np.dot(self.layer1 , self.weight2)
        self.layer2   =  sigmod(b)
        c = np.dot(self.layer2 ,self.weight3)
        self.output = sigmod(c)

        return  self.output

    def  backprop (self):

        #caculate  derivative of the loss function with respect to a_weight3 , a_weight2 and a_weight1
        a_weight3  =  np.dot(self.layer2.T  , 2*(self.y -  self.output)*calculate_z(self.output))
        a_weight2  =  np.dot(self.layer1.T,  (np.dot(2*(self.y - self.output) * calculate_z(self.output), self.weight3.T) * calculate_z(self.layer2)))
        a_weight1 = np.dot(self.input.T,  (np.dot(np.dot(2*(self.y - self.output) * calculate_z(self.output), self.weight3.T),self.weight2) * calculate_z(self.layer1)))
        #update  weight1 anh weight2 weight3
        self.weight3  += a_weight3
        self.weight2  += a_weight2
        self.weight1  +=a_weight1

    def train ( self , x , y):
        self.feedforward()
        self.backprop()

    def Predict (self ,  x):
        a  = np.dot(x, self.weight1)
        self.layer1  =  sigmod(a)
        b = np.dot(self.layer1 , self.weight2)
        self.layer2   =  sigmod(b)
        c = np.dot(self.layer2 ,self.weight3)
        self.output = sigmod(c)

        return  self.output

NN  = Neural_network(x , y)
for  i in range (1500):
    if i % 100 == 0 :
        print( " for  iteration : " ,str(i) ,"\n")
        print( " input : \n ", x )
        print("Actual Output: \n" ,y)
        print("Predicted Output: \n",NN.feedforward())
        print("Loss: \n" + str(np.mean(np.square(y - NN.feedforward()))))  # mean sum squared loss
        print("\n")
    NN.train(x, y)

print("\n")
print("W 1 :", NN.weight1.T)
print("W 2 :", NN.weight2.T)
print( "Perdict {2.1} : ",NN.Predict([2.1]))
