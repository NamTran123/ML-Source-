import  numpy as  np
import  matplotlib.pyplot  as  plt
#
# x=np.array(([0,0,1,1],[0,1,1,1],[1,0,1,1],[1,1,1,1]), dtype=float)
# y=np.array(([0],[1],[0],[1]), dtype=float)
# x = np.array(([0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]))
# print(x.shape)
# y = np.array([[0, 1, 1, 0]])
#x=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
#y=np.array(([0],[1],[1],[0]), dtype=float)
# x=np.array(([0,0,1,1],[0,1,1,1],[1,0,1,1],[1,1,1,1],[0,0,0,1],[0,0,1,0],[1,0,0,1],[1,1,0,0]), dtype=float)
# y=np.array(([0],[1],[1],[1],[0],[0],[0],[0]), dtype=float)

x=np.array(([0,0,1,1],[0,1,1,1],[1,0,1,1],[1,1,1,1],[0,0,0,1],[0,0,1,0],[1,0,0,1],[1,1,0,0]), dtype=float)
y=np.array(([0],[1],[1],[1],[0],[0],[0],[0]), dtype=float)



def sigmod(x):
    return (1 / (1 + np.exp(-x)))

def calculate_z(z):
    return  (z*(1-z))

class  Neural_network:
    def  __init__(self ,x ,y):
        self.input  = x
        self.weight1  =  np.random.rand(self.input.shape[1] ,8)
        self.weight2  =  np.random.rand(8,8)
        self.weight3  =  np.random.rand(8,1)
        self.y = y
        self.output  =  np.zeros(y.shape)

    def feedforward (self ):
        a  = np.dot(self.input, self.weight1)
        self.layer1  =  sigmod(a)

        c  =  np.dot(self.layer1 , self.weight2 )
        self.layer2  = sigmod(c)


        b = np.dot(self.layer2 , self.weight3)
        self.output  =  sigmod(b)
        return  self.output

    def  backprop (self):

        #caculate  derivative of the loss function with respect to weights2 and weights1
        a_wright3 =  np.dot(self.layer2.T  , 2*(self.y -  self.output)*calculate_z(self.output))
        a_wright2 =  np.dot(self.layer1.T ,a_wright3 * calculate_z(self.layer2) )
        a_wright1 =  np.dot(self.input.T ,a_wright2 * calculate_z(self.layer1))

        #update  weight1 anh weight2
        self.weight3  += a_wright3
        self.weight2  +=  a_wright2
        self.weight1  +=  a_wright1

    def train ( self , x , y):
        self.feedforward()
        self.backprop()

NN  = Neural_network(x , y)
for  i in range (2000):
    if i % 100 ==0 :
        print( " for  iteration : " ,str(i) ,"\n")
        print( " input : \n ", str(x) )
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str(NN.feedforward()))
        print("Loss: \n" + str(np.mean(np.square(y - NN.feedforward()))))  # mean sum squared loss
        print("\n")
    NN.train(x, y)



