import numpy as np
import cv2
import glob

count = 1
cam = cv2.VideoCapture(1)
while(1):
    f,img = cam.read()
    
    cv2.imshow('img',img)
    key = cv2.waitKey(5) % 0x100
    if (key == 10):
        name = "imagen_fondo.jpg"
        count += 1
        #print name
        cv2.imwrite(name, img)
    if key == 27:
        break

cv2.destroyAllWindows()
