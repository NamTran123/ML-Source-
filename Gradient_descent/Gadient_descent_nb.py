# # import  numpy as  np  
# # import  math   
# # np.random.seed(2)
# # # # height (cm)
# # # x = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# # # # weight (kg)
# # # y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
# # x = np.random.rand(1000, 1)
# # y = 4 + 3 * x + .2*np.random.randn(1000, 1) # noise added
# # #Xay dung X mo rong  (add A + [1....X.shape[0])
# # one =  np.ones((x.shape[0] ,1))

# # Xbar   = np.concatenate((one,x) , axis =1) 
# # # Tinh gia tri dao ham 

# # def  tinh_gia_tri_dao_ham(w):
# #     N = Xbar.shape[0]
# #     return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

# # def  tinh_gia_tri_ham_so(w):
# #     N = Xbar.shape[0]
# #     return (.5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2)

# # def  linear_regression(eta , w_start ,tinh_gia_tri_dao_ham ):
# #     w = [w_start]
# #     for it in range(100):
# #         w_new = w[-1] - eta*tinh_gia_tri_dao_ham(w[-1])
# #         if np.linalg.norm(tinh_gia_tri_dao_ham(w_new))/len(w_new) < 10**-9:
# #             break 
# #         w.append(w_new)
# #     return (w, it) 

# # w_init = np.array([[2], [2]])
# # (w, it1) = linear_regression(1,w_init, tinh_gia_tri_dao_ham)
# # print( w[-1].T)
# # print (tinh_gia_tri_ham_so(w[-1]))
# import numpy as np 
# import matplotlib
# import matplotlib.pyplot as plt
# np.random.seed(2)

# X = np.random.rand(1000, 1)
# y = 4 + 3 * X + .2*np.random.randn(1000, 1) # noise added

# # Building Xbar 
# one = np.ones((X.shape[0],1))
# Xbar = np.concatenate((one, X), axis = 1)
# A = np.dot(Xbar.T, Xbar)
# b = np.dot(Xbar.T, y)
# w_lr = np.dot(np.linalg.pinv(A), b)
# print('Solution found by formula: w = ',w_lr.T)

# # Display result
# w = w_lr
# w_0 = w[0][0]
# w_1 = w[1][0]
# x0 = np.linspace(0, 1, 2, endpoint=True)
# y0 = w_0 + w_1*x0

# # Draw the fitting line 
# plt.plot(X.T, y.T, 'b.')     # data 
# plt.plot(x0, y0, 'y', linewidth = 2)   # the fitting line
# plt.axis([0, 1, 0, 10])
# plt.show()

# def grad(w):
#     N = Xbar.shape[0]
#     return (1/N * Xbar.T.dot(Xbar.dot(w) - y))

# def cost(w):
#     N = Xbar.shape[0]
#     return (.5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2)

# def numerical_grad(w, cost):
#     eps = 1e-4
#     g = np.zeros_like(w)
#     for i in range(len(w)):
#         w_p = w.copy()
#         w_n = w.copy()
#         w_p[i] += eps 
#         w_n[i] -= eps
#         g[i] = (cost(w_p) - cost(w_n))/(2*eps)
#     return g 

# def check_grad(w, cost, grad):
#     w = np.random.rand(w.shape[0], w.shape[1])
#     grad1 = grad(w)
#     grad2 = numerical_grad(w, cost)
#     return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False 

# print( 'Checking gradient...', check_grad(np.random.rand(2, 1), cost, grad))


# def myGD(w_init, grad, eta):
#     w = [w_init]
#     for it in range(100):
#         w_new = w[-1] - eta*grad(w[-1])
#         if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
#             break 
#         w = w.append(w_new)
#     return (w, it) 

# w_init = np.array([[2], [1]])
# (w1, it1) = myGD(w_init, grad, .1)
# print( w1[-1].T, it1+1)
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2*np.random.randn(1000, 1) # noise added

one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)

def grad(w):
    N = Xbar.shape[0]
    return (1/N * Xbar.T.dot(Xbar.dot(w) - y))

def myGD(w_init, eta):
    w = w_init
    for it in range(10000):
        w = w - eta*grad(w)
    return w

w_init = np.array([[2], [1]])
w = myGD(w_init, 0.1)
print(w)