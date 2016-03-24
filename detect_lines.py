import cv2
import numpy as np
import math

def dibujar_rectangulos(box,im):
    box1=len(box)
    c=0
    
    while c!=box1:                                 #
        a=box[c]                                   #
        array = np.int0(a)                         #
        cv2.drawContours(im,[array],0,(0,255,0),2) #
        c=c+1    

def ordenar_rectangulos(rectangulos):
    rectangulos.sort()
    return rectangulos


def encontrar_rectangulos_perspective(final,im):

    vacantes = 0
    width = 300
    height = 300
    linea=final
    lista=[]
    rectangulos = []
    rectangulos_ordenados = []
    region_interes = []
    region_ordenada = []
    """first = linea[3]
    lista.append(first)
    #linea.remove(first)
    cv2.line(im,(first[0],first[1]),(first[0],first[1]),(0,255,0),10)
    cv2.line(im,(first[0],first[1]),(first[2],first[3]),(0,0,0),4)   """
    
    while len(linea)>0:
        first=linea[0]
        linea.remove(first)
        comparadas=linea
        punto_medio_first_x = (first[2]+first[0])/2
        punto_medio_first_y = (first[3]+first[1])/2
        for x1,y1,x2,y2 in comparadas:
            punto_medio_aux_x = (x1+x2)/2
            punto_medio_aux_y = (y2+y1)/2
            distancia = math.sqrt ((punto_medio_aux_x  -  punto_medio_first_x)**2 + (punto_medio_aux_y - punto_medio_first_y)**2)
            
            if distancia < 200:
                
               
                
                punto_medio_x=(first[0]+x1)/2
                punto_medio_y=(first[1]+y1)/2
                
                punto_medio_x1=(first[2]+x2)/2
                punto_medio_y1=(first[3]+y2)/2

                punto_medio_xh=punto_medio_x+10
                
                
                punto_medio_x1h=punto_medio_x1+10
               
                rectangulo=((first[0],first[1]),(x1,y1),(x2,y2),(first[2],first[3]))
                
                rectangulos.append(rectangulo)

	rectangulos_ordenados = ordenar_rectangulos(rectangulos)
    #print len(rectangulos_ordenados)
    return rectangulos_ordenados


def filtrar_lineas_perspective1(lines,im):
    #.....Declaracion de arrays auxiliares
    final = []
    lista = lines
    rectangulos_perspective = []

    #......Ciclo While hasta filtrar todas las lineas
    while len(lista)>0:
        #.....vector del grupo de lineas detectectada por cada franja de linea blanca en un puesto de estacionamiento
        linea= []
        #.....variables con valores para determinar el alto y bajo de una linea
        ymayor=0
        ymenor=20000
        xmayor=0
        xmenor=20000
        
        #.....primera linea del vector 
        first = lista[0]
        #.....agregamos esa primera linea al vector linea
        linea.append(first)
        #.....removemos esa linea del vector lista para no comparar nuevamente con ella en la proxima iteracion
        lista.remove(first)
        #.....calculamos el punto medio de la linea first
        punto_medio_first_x = (first[2]+first[0])/2
        punto_medio_first_y = (first[3]+first[1])/2
        #.....comparamos la distancia de esa linea con las otras del vector lista
        for x1,y1,x2,y2 in lista:
            #.....calculamos el punto medio de la linea a comparar
            punto_medio_aux_x = (x1+x2)/2
            punto_medio_aux_y = (y2+y1)/2
            #.....si la distancia de la linea first a otra es menor de  pixeles quiere decir que pertenece a una franja de una linea blanca 
            if (math.sqrt ((punto_medio_aux_x  -  punto_medio_first_x)**2 + (punto_medio_aux_y - punto_medio_first_y)**2))< 100:
                #.....ese elemento se agrega al vector linea
                linea.append([x1,y1,x2,y2])
                #.....se remueve del vector lista para no iterar con el
                lista.remove([x1,y1,x2,y2])
                
        for x1,y1,x2,y2 in linea:
            if y1>y2:
                if y1>ymayor:
                    ymayor=y1
                    xmayor=x1
            if y2>y1:
                if y2>ymayor:
                    ymayor=y2
                    xmayor=x2
            if y1<y2:
                if y1<ymenor:
                    ymenor=y1
                    xmenor=x1
            if y2<y1:
                if y2<ymenor:
                    ymenor=y2
                    xmenor=x2
            
                
            

        #.....las lineas mas largas detectadas se agregan a un vector final            
        final.append([xmenor,ymenor,xmayor,ymayor])
        #.....Se recorren las lineas del vector final y se pinta cada linea en la imagen
    #x1,y1,x2,y2 = final[4]
    #final.remove([x1,y1,x2,y2])
    #cv2.line(im,(x1,y1),(x2,y2),(0,255,0),2)
    #x1,y1,x2,y2 = final[6]
    #final.remove([x1,y1,x2,y2])
    #cv2.line(im,(x1,y1),(x2,y2),(0,255,0),2)
    #x1,y1,x2,y2 = final[5]
    #final.remove([x1,y1,x2,y2])
    print len(final)
    for x1,y1,x2,y2 in final:
        cv2.line(im,(x1,y1),(x2,y2),(0,255,0),2)
    rectangulos_perspective = encontrar_rectangulos_perspective(final,im)
    return rectangulos_perspective
    
    
    

######################################################Funcion principal######################################################
img1 = cv2.imread('img_perspective1.jpg')
#img2 = cv2.imread('imagen/img_perspective2.jpg')
c = 0

#.......Rango de color blanco
sensitivity = 50
lower_white = np.array([0,0,255-sensitivity])
upper_white = np.array([255,255+sensitivity,255])


hsv_perspective1     = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
mask_perspective1    = cv2.inRange(hsv_perspective1, lower_white, upper_white)
        

flag,b_perspective1 = cv2.threshold(mask_perspective1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
element             = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))          
cv2.erode(b_perspective1,element)



lines_perspective1  = cv2.HoughLinesP(b_perspective1,1,np.pi/720.0,80,100,210)[0].tolist()
'''for x1,y1,x2,y2 in lines_perspective1:
    cv2.line(img1,(x1,y1),(x2,y2),(0,255,0),2)'''

rectangulos1        = filtrar_lineas_perspective1(lines_perspective1,img1)
dibujar_rectangulos(rectangulos1,img1)

'''
for x1,y1,x2,y2 in lines_perspective1:
    cv2.line(img1,(x1,y1),(x2,y2),(0,255,0),1)

rectangulos1 = filtrar_lineas_perspective1(lines_perspective1,img1)
dibujar_rectangulos(rectangulos1,img1,vacantes1)

lines_perspective2 = cv2.HoughLinesP(b_perspective2,1,np.pi/720.0,100,100,90)[0].tolist()
#for x1,y1,x2,y2 in lines_perspective2:
#    cv2.line(img2,(x1,y1),(x2,y2),(0,255,0),1)
rectangulos2, vacantes2 = filtrar_lineas_perspective2(lines_perspective2,img2)
dibujar_rectangulos(rectangulos2,img2,vacantes2)

dimensiones1 = dimension(rectangulos1)
dimensiones2 = dimension(rectangulos2)


images_roi1 = extraer_imagen_roi(rectangulos1,dimensiones1,img1)
images_roi2 = extraer_imagen_roi(rectangulos2,dimensiones2,img2)'''

#cv2.imshow('image2',b_perspective1)
cv2.imshow('image1',img1)
cv2.waitKey(0)
cv2.destroyAllWindows() 