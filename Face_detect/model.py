import numpy as  np 
from PIL  import  Image
import  glob as  gl  
import  os

D = 165*120 # original dimension

def build_list_filename(): 
    for filename in os.listdir('/home/sink-all/Desktop/ML Source/Face_detect/nottingham/'): 
        listfilename = []
        listfilename.append(filename)
        return listfilename

def  vetor_image(image):
    return image.reshape(1,D)

main()






