import cv2
import numpy as np
import math
import db_conexion
import psycopg2


#..............................................................................................................................#
class StatModel(object):
    ''' INICIO DE LA MAQUINA DE VECTOR DE SOPORTE'''    
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)
#..............................................................................................................................#


#..............................................................................................................................#
class SVM(StatModel):
    '''ENTRENAR Y PREDECIR MUESTRAS'''
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses):
        #setting algorithm parameters
        params = dict( kernel_type = cv2.SVM_LINEAR, 
                       svm_type = cv2.SVM_C_SVC,
                       C = 1 )
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(samples)])
#..............................................................................................................................#


#CREAR ZONAS EN LA BASE DE DATOS#
#..............................................................................................................................#
def crear_zonas_bd(lista):
    con      =  db_conexion.conexion()
    cur      = con.cursor()
    iterator = 1
    for i in lista:
        try:
            cur.execute("INSERT INTO zona(estado,id_estacionamiento,descripcion) VALUES (%s,%s,%s) RETURNING id_zona",(0,2,'Zona ' +str(iterator)+' Del Estacionamiento 1'))       
            id_zona = cur.fetchone()[0]
            
            for puestos in i:
                try:
                    cur.execute("INSERT INTO puesto_estacionamiento(id_zona,estado) VALUES (%s,%s)",(str(id_zona),0))
                except:
                    print 'NO SE PUEDE INSERTAR EN PUESTOS'

        except psycopg2.Error as e:
            pass
            print 'NO SE PUEDE INSERTAR EN ZONA'
        iterator += 1          
        con.commit()
  
#..............................................................................................................................#

#OBTENER ID DE LOS PUESTOS DE ESTACIONAMIENTO DE LA BASE DE DATOS
#..............................................................................................................................#
def obtener_id_puestos(lista):
    con           =  db_conexion.conexion()
    cur           = con.cursor()
    zonas         = []
    lista_id_puestos = []
    puesto        = []
    try:
        cur.execute("SELECT id_zona from zona where id_estacionamiento= 2 order by id_zona asc")
    except:
        print "I can't SELECT from zona"

    rows = cur.fetchall()
    for row in rows:
        zonas.append(row[0])
        
    for zon in zonas:
        try:
            cur.execute("SELECT id_puesto from puesto_estacionamiento where id_zona = %s order by id_puesto asc",(str(zon),))
            
            rows = cur.fetchall()
            for row in rows:
                puesto.append(row[0])
            lista_id_puestos.append(puesto)
            puesto = []

        except:
            print "I can't SELECT from puesto"

    return lista_id_puestos

#..............................................................................................................................#

# CREA LISTA DE ZONAS Y PUESTOS DE ESTACIONAMIENTO CON LOS ROIS DETECTADOS
#..............................................................................................................................#
def iniciacion_estacionamiento(varianza, promedio, rectangulos):
    vp_zonas      = []
    lista_zonas   = []
    counter       = 0

    for iterator in promedio:
        if counter == 0:
            zona = []
            zona.append(rectangulos[counter])
            lista_zonas.append(zona)
            vp_zonas.append(iterator)
        else:
            aux_pointer = 0
            for i in vp_zonas:
                if iterator < (i+varianza) and iterator > (i-varianza):
                    lista_zonas[aux_pointer].append(rectangulos[counter])
                    vp_zonas[aux_pointer] = (vp_zonas[aux_pointer] + iterator) / 2
                    break
                aux_pointer += 1
            if aux_pointer == len(vp_zonas):
                zona = []
                zona.append(rectangulos[counter])
                lista_zonas.append(zona)
                vp_zonas.append(iterator)
        counter += 1
    return lista_zonas
#..............................................................................................................................#

# CALCULO DE FUNCIONES PARA CREAR LAS ZONAS
#..............................................................................................................................# 
def calcular_varianza(rectangulos):
    
    dimensiones =[]
    f = np.int0(rectangulos)
    c = 0
    while c<len(rectangulos):
        
        lado_1 = f[c]
        h_linea_1 = lado_1[3]
        h_linea_2 = lado_1[0]
        h_linea_1_x = h_linea_1[0]
        h_linea_1_y = h_linea_1[1]
        h_linea_2_x = h_linea_2[0]
        h_linea_2_y = h_linea_2[1]
        altura_linea_1 = math.sqrt ((h_linea_2_x  -  h_linea_1_x)**2 + (h_linea_2_y - h_linea_1_y)**2)
       
        h_linea_1 = lado_1[1]
        h_linea_2 = lado_1[2]
        h_linea_1_x = h_linea_1[0]
        h_linea_1_y = h_linea_1[1]
        h_linea_2_x = h_linea_2[0]
        h_linea_2_y = h_linea_2[1]
        altura_linea_2 = math.sqrt ((h_linea_2_x  -  h_linea_1_x)**2 + (h_linea_2_y - h_linea_1_y)**2)
        
        if altura_linea_1<altura_linea_2:
            altura = altura_linea_2
        else:
            altura = altura_linea_1

        dimensiones.append(int(altura))
               
        c=c+1
    varianza = sum(dimensiones) / len(dimensiones)
    return varianza/2
#..............................................................................................................................#

# CALCULO DE FUNCIONES PARA CREAR LAS ZONAS

#..............................................................................................................................#  
def calcular_promedio(rectangulos):
    #print rectangulos
    primero = []
    promedio =[]
    eyes      =[]
    sum_eyes = []
    c  = 0
    c1 = 0
    for x,y,z,w in rectangulos:
       prom = (x[1]+y[1]+z[1]+w[1]) / 4
       promedio.append(prom)

    return promedio       
#..............................................................................................................................#

#REDIMENSIONAR LAS ROIS
#..............................................................................................................................#
def rectificar_roi(lista):
    images_rect       = []
    lista_images_rect = []

    for zonas in lista:
        for puestos in zonas:
            resized_image = cv2.resize(puestos,(150,150))
            images_rect.append(resized_image)

        lista_images_rect.append(images_rect)
        images_rect = []

    return lista_images_rect
#..............................................................................................................................#

#FUNCIN QUE INDICA SI EL PUESTO ESTA OCUPADO Y ACTUALIZA BD
#..............................................................................................................................#
def determinar_ocupado(images_roi_binari,images_roi,frame_perspective,lista,lista_id_puestos,lista_circulos,frame):
    um                 = 500
    aux_pointer_zona   = 0
    aux_pointer_puesto = 0
    con                =  db_conexion.conexion()
    cur                = con.cursor()

    for zonas in lista:
        for puestos in zonas:
            roi_binari          = images_roi_binari[aux_pointer_zona][aux_pointer_puesto]
            roi                 = images_roi[aux_pointer_zona][aux_pointer_puesto]
            coordenadas         = puestos
            id_puesto           = lista_id_puestos[aux_pointer_zona][aux_pointer_puesto]
            contours, hierarchy = cv2.findContours(roi_binari,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                cv2.fillPoly(roi, contours, (0,0,0))
                treshold    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                total       = treshold.size
                no_zero     = cv2.countNonZero(treshold)
                pix_negros  = total - no_zero

                if  pix_negros>(no_zero/2):
                    a       = coordenadas
                    array   = np.int0(a)                         
                    cv2.drawContours(frame_perspective,[array],0,(0,0,255),2) #
                    cv2.circle(frame,lista_circulos[aux_pointer_zona][aux_pointer_puesto], 20, (0,0,255), -1)
                    try:
                        cur.execute("SELECT estado FROM puesto_estacionamiento WHERE id_puesto = %s", (str(id_puesto),))
                        rows = cur.fetchall()
                        for row in rows:
                            estado = row[0]
                        if estado == 0:
                            try:
                                cur.execute("UPDATE puesto_estacionamiento SET estado = %s WHERE id_puesto = %s", (str(1),str(id_puesto),))
                                con.commit()
                                print 'ACTUALIZO'
                            except:
                                print 'NO SE PUEDE ACTUALIZAR ESTADO DE PUESTOS DE ESTACIONAMIENTO'
                    except:
                        print 'NO SE PUEDE SELECCIONAR ESTADO DEL PUESTO DE ESTACIONAMIENTO'   
                else:
                    try:
                        cur.execute("SELECT estado FROM puesto_estacionamiento WHERE id_puesto = %s", (str(id_puesto),))
                        rows = cur.fetchall()
                        for row in rows:
                            estado = row[0]
                        if estado == 1:
                            print estado
                            try:
                                cur.execute("UPDATE puesto_estacionamiento SET estado = %s WHERE id_puesto = %s", (str(0),str(id_puesto),))
                                con.commit()
                            except:
                                print 'NO SE PUEDE ACTUALIZAR ESTADO DE ESTE PUESTO DE ESTACIONAMIENTO' 
                    except:
                        print 'NO SE PUEDE SELECCIONAR ESTADO DEL PUESTO DE ESTACIONAMIENTO'   
                    
                    
            aux_pointer_puesto += 1

        aux_pointer_zona   += 1
        aux_pointer_puesto  = 0
#..............................................................................................................................#

#EXTRACCION DE ROIS
#..............................................................................................................................#
def extraer_imagen_roi(lista,dimensiones,im):
    
    images              = []
    lista_images        = []
    aux_pointer_zona    = 0
    aux_pointer_puesto  = 0

    for zonas in lista:
        for puestos in zonas:

            puesto_array            = np.int0(puestos)
            coor_puesto_array       = np.array(puesto_array, dtype = "float32")

            dimension_array         = dimensiones[aux_pointer_zona][aux_pointer_puesto]
            coor_dimension_array    = np.array([[0,0],[dimension_array[1],0],[dimension_array[1],dimension_array[0]],[0,dimension_array[0]]],np.float32)

            mask = np.zeros(im.shape,dtype = np.uint8)
            cv2.fillPoly(mask,[puesto_array],(255,255,255))
            masked_image = cv2.bitwise_and(im,mask)

            M = cv2.getPerspectiveTransform(coor_puesto_array,coor_dimension_array)
            roi = cv2.warpPerspective(masked_image,M,(dimension_array[1],dimension_array[0]))

            images.append(roi)
            aux_pointer_puesto += 1
        lista_images.append(images)
        images = []
        aux_pointer_zona   += 1
        aux_pointer_puesto  = 0

    return lista_images
#..............................................................................................................................#

# CALCULO DE FUNCIONES PARA CREAR LOS ROIS
#..............................................................................................................................#
def dimension(lista):
    dimensiones         = []
    lista_dimensiones   = []
    
    for zonas in lista:
        for puestos in zonas:
            
            lado_1          = puestos
            coor_lin_h_1    = lado_1[3]
            coor_lin_h_2    = lado_1[0]
            coor_lin_1_h_x  = coor_lin_h_1[0]
            coor_lin_1_h_y  = coor_lin_h_1[1]
            coor_lin_2_h_x  = coor_lin_h_2[0]
            coor_lin_2_h_y  = coor_lin_h_2[1]
            altura_linea_1  = math.sqrt ((coor_lin_2_h_x  -  coor_lin_1_h_x)**2 + (coor_lin_2_h_y - coor_lin_1_h_y)**2)

            lado_1          = puestos
            coor_lin_h_1    = lado_1[1]
            coor_lin_h_2    = lado_1[2]
            coor_lin_1_h_x  = coor_lin_h_1[0]
            coor_lin_1_h_y  = coor_lin_h_1[1]
            coor_lin_2_h_x  = coor_lin_h_2[0]
            coor_lin_2_h_y  = coor_lin_h_2[1]
            altura_linea_2  = math.sqrt ((coor_lin_2_h_x  -  coor_lin_1_h_x)**2 + (coor_lin_2_h_y - coor_lin_1_h_y)**2)

            if altura_linea_1 < altura_linea_2:
                altura = altura_linea_2
            else:
                altura = altura_linea_1
            
            coor_lin_w_1    = lado_1[0]
            coor_lin_w_2    = lado_1[1]
            coor_lin_1_w_x  = coor_lin_w_1[0]
            coor_lin_1_w_y  = coor_lin_w_1[1]
            coor_lin_2_w_x  = coor_lin_w_2[0]
            coor_lin_2_w_y  = coor_lin_w_2[1]
            ancho_linea_1   = math.sqrt ((coor_lin_2_w_x  -  coor_lin_1_w_x)**2 + (coor_lin_2_w_y - coor_lin_1_w_y)**2)
       
            coor_lin_w_1    = lado_1[3]
            coor_lin_w_2    = lado_1[2]
            coor_lin_1_w_x  = coor_lin_w_1[0]
            coor_lin_1_w_y  = coor_lin_w_1[1]
            coor_lin_2_w_x  = coor_lin_w_2[0]
            coor_lin_2_w_y  = coor_lin_w_2[1]
            ancho_linea_2   = math.sqrt ((coor_lin_2_w_x  -  coor_lin_1_w_x)**2 + (coor_lin_2_w_y - coor_lin_1_w_y)**2)

            if ancho_linea_1 < ancho_linea_2:
                ancho = ancho_linea_2
            else:
                ancho = ancho_linea_1

            dimensiones.append((int(altura),int(ancho)))

        
        lista_dimensiones.append(dimensiones)
        dimensiones = []
        
    return lista_dimensiones
#..............................................................................................................................#


#..............................................................................................................................#
def ordenar_rectangulos(rectangulos):
    rectangulos.sort()
    return rectangulos
#..............................................................................................................................#

#ENCNTRAR RECTACNGULOS CON LAS LINEAS DETECTADAS
#..............................................................................................................................#
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
    
    
    return rectangulos_ordenados
#..............................................................................................................................#


#..............................................................................................................................#
def mostrar_text(im,vacantes):                                                                                 
    cv2.putText(im,'Numero de puestos en el estacionamiento %d.'% vacantes,(110,20), cv2.FONT_HERSHEY_PLAIN, 1.0,(0,0,0))  
#..............................................................................................................................#


#..............................................................................................................................#    
def dibujar_rectangulos(box,im):          
    
    for zonas in box:
        for puestos in zonas:
            a = puestos
            array = np.int0(a)
            cv2.drawContours(im,[array],0,(0,255,0),2)
#..............................................................................................................................#

#DETECTAR LINEAS
#..............................................................................................................................#
def filtrar_lineas_perspective(lines,im):
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
    

    rectangulos_perspective = encontrar_rectangulos_perspective(final,im)
    return rectangulos_perspective

#..............................................................................................................................#


#.......................................................................#
def four_point_transform(frame, pts1,pts2):                             #
    M       = cv2.getPerspectiveTransform(pts1, pts2)                          #
    warped  = cv2.warpPerspective(frame, M, (720, 480))                  #
    return warped                                                       #
#.......................................................................#


#################################################################################################################################
cam         = cv2.VideoCapture(1)
img_fondo   = cv2.imread('imagen_fondo.jpg')
img1        = cv2.imread('img_perspective.jpg')


#.......Rango de color blanco perspective 1 
sensitivity = 50
lower_white = np.array([0,0,255-sensitivity])
upper_white = np.array([255,255+sensitivity,255])

#.......Coordenadas perspective 1
pts1 = np.float32([[0,180],[580,180],[0,310],[582,315]])
pts2 = np.float32([[0,0],[720,0],[0,480],[720,480]])

lista_circulos = []
lista_zonas = []
id_puestos  = []
rectangulos = []
dimensiones = []
circulos    = []
images_roi  = []
counter     = 0

hsv_perspective1     = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
mask_perspective1    = cv2.inRange(hsv_perspective1, lower_white, upper_white)

#............................
flag,b_perspective1  = cv2.threshold(mask_perspective1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
element              = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))          
cv2.erode(b_perspective1,element)

#............................
lines_perspective1  = cv2.HoughLinesP(b_perspective1,1,np.pi/720.0,80,100,210)[0].tolist()
rectangulos         = filtrar_lineas_perspective(lines_perspective1,img1)


#dimensiones1 = dimension(rectangulos1)
#dibujar_rectangulos1(lista_zonas,img1)


varianza             = calcular_varianza(rectangulos)
promedio             = calcular_promedio(rectangulos)
lista_zonas         = iniciacion_estacionamiento(varianza, promedio,rectangulos)

#crear_zonas_bd(lista_zonas)
dimensiones2        = dimension(lista_zonas)
id_puestos          = obtener_id_puestos(lista_zonas)


zona_1      = [(150,324),(325,324),(508,324)]
#zona_2      = [(390,127),(448,133),(510,144)]
#lista_circulos = []
lista_circulos.append(zona_1)


fgbg = cv2.BackgroundSubtractorMOG()

while True:
    counter += 1;
    frame           = cam.read()[1]
    
    
    if counter %30 == 0 and frame.shape!=None :
        
        
        fgmask  = fgbg.apply(frame)
        kernel  = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))    
        gray    = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        frame_perspective          = four_point_transform(frame,pts1,pts2)
        frame_perspective_binari   = four_point_transform(gray, pts1,pts2)

    
        dibujar_rectangulos(lista_zonas,frame_perspective)
        for zona in lista_circulos:
            for coor in zona:
                cv2.circle(frame,coor, 20, (0,255,0), -1)
                
       
        images_roi2          = extraer_imagen_roi(lista_zonas,dimensiones2,frame_perspective)
        images_roi_binari2   = extraer_imagen_roi(lista_zonas,dimensiones2,frame_perspective_binari)

        images_roi2          = rectificar_roi(images_roi2)
        images_roi_binari2   = rectificar_roi(images_roi_binari2)


        determinar_ocupado(images_roi_binari2,images_roi2,frame_perspective,lista_zonas,id_puestos,lista_circulos,frame)

        cv2.imshow("ventana1", frame_perspective)
        cv2.imshow("ventana2", frame)
        

        if cv2.waitKey(5) & 0xFF == 27:
            break

cam.release()
cv2.destroyAllWindows() 
