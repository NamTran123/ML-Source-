import  numpy as  np  

#Zero -dimensional  arr

X  = np.array(10)
print('X :',X)

print(type(X))
print(np.ndim(X))

#One- dimensional arr

X =  np.array(
    [1,3,4,56,7,6,3])
Y  =  np.array([1.2,23.5,124,1,4.5,6,7.9])
print('X :',X)
print('Y :' ,Y)
print('Dimensional X:', np.ndim(X))
print('Dimensional Y:', np.ndim(Y))
print(len(X))

#Two dimensional  arr
X =  np.array([ 
    [1,2,34,5,56,67],
    [23,43,2,12,23,12],
    [1,12,32,4,1,21]
])

print(np.ndim(X)) #2

print(X[0])
print(X[1])
print(X[2])
print(X)

#Three mimensional  arr
X  = np.array([
    [[12,13],[12,24]],
    [[43,45],[12,45]],
    [[12.34,233],[47,787]]
])
print(X)
print(np.ndim(X))
print(X[1])