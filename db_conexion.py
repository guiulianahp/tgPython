import psycopg2
import sys


def conexion():
    con=None
    try:
     
        con = psycopg2.connect(database='SEI_VC', user='guiuliana')  
    
    except psycopg2.DatabaseError, e:
        print 'Error %s' % e    
        sys.exit(1)
    
    return con

con = conexion()
con.close()

