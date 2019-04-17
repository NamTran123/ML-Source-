import  numpy as  np 

# Create arr  from list or tuple
a = list([1,3,4,5])
x  =  np.array(a)
print(x)
b  =  tuple([1,2,3,4,5,5,6])
x = np.array(b)
print(x)

#Create  list using arange()
arr  =  np.arange(1,10) #[1,2,3,4,5,6,7,8,9]
print(arr)

arr  =  np.arange(1.2 ,10.5 ,2)
print(arr)

arr  = np.arange(1 ,10 ,2 ,int)
print(arr)

#Create list  using  linspace  
#Create 50 point in linspace from 1 to 10  
arr1  =  np.linspace(1,10) # Define  50 point 
print(arr1)

arr1  =  np.linspace(1,100  ,7) #7 point  in linspace   (1->100)
print(arr1)

arr1  = np.linspace(1,100 ,endpoint = False) #excluding the endpoint