import  cv2  
import  numpy  as np  

image =  cv2.imread("/home/sink-all/Desktop/ML Source/image.jpg")
cv2.imshow("image1" , image)
img  =  image[200: 330,0 : 600]
print (image.shape)
cv2.imshow("image2" , img)
img2  =  cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
cv2.imshow("image3" , img2) 
cv2.waitKey()