import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

img = cv2.imread('imagen_fondo.jpg')
rows,cols,ch = img.shape

# Medidas dinamicas
pts1 = np.float32([[0,180],[580,180],[0,310],[582,315]])
pts2 = np.float32([[0,0],[720,0],[0,480],[720,480]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(720,480))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()


cv2.imshow('Image',dst)
cv2.imwrite('img_perspective.jpg', dst)
cv2.waitKey(0)
cv2.destroyAllWindows() 
