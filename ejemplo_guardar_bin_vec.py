# -*- coding: utf-8 -*-

import cPickle as pickle #en python 3 es directo pickle
import numpy as np

def init_file(filename,header_dict):
    fid=open(filename,'wb')
    pickle.dump(header_dict,fid,pickle.HIGHEST_PROTOCOL)
    return fid

def save_bin_vector(fid,np_array):#asume fid abierto como wb y np_array un array de numpy. Sirve para vectores y matrices
    #fid.write(np_array.tobytes())
    np.save(fid,np_array)
    
def save_vector_extra(fid,np_array,extra):#también se puede guardar con más info
    tmp={'Data':np_array,'Extra': extra}
    pickle.dump(tmp,fid,pickle.HIGHEST_PROTOCOL)

#%%Ejemplo guardar vector
    
orig_matrix=np.random.random((100,100)) #Matriz original

header={'Filas matriz original':orig_matrix.shape[0],'Columnas matriz original':orig_matrix.shape[1]} #Header de ejemplo, puede tener cualquier cosa _binarizable_ por pickle

vector=orig_matrix[0,:] #vector de ejemplo para guardar

filename='example_vect.bin'
fid=init_file(filename,header)
save_bin_vector(fid,vector)
fid.close()


#%% Ejemplo lectura

with open('example_vect.bin',"rb") as fid:
    header=pickle.load(fid) #leo header
    print header
    read_vector=np.load(fid) #leo vector
    print "Maxima diferencia entre vector original y vector leido: ", np.max(np.abs(vector-read_vector))


#%%Ejemplo guardado con info extra
    
orig_matrix=np.random.random((100,100)) #Matriz original

header={'Filas matriz original':orig_matrix.shape[0],'Columnas matriz original':orig_matrix.shape[1]} #Header de ejemplo, puede tener cualquier cosa _binarizable_ por pickle

vector=orig_matrix[0,:] #vector de ejemplo para guardar

filename='example_vect_extra.bin'
fid=init_file(filename,header)
save_vector_extra(fid,vector, 'string de ejemplo')
fid.close()

#%% Ejemplo lectura

with open('example_vect_extra.bin',"rb") as fid:
    header=pickle.load(fid) #leo header
    print header
    vector_extra=pickle.load(fid) #leo vector con data extra
    print "Maxima diferencia entre vector original y vector leido: ", np.max(np.abs(vector-vector_extra['Data']))
    print "Data extra: ", vector_extra['Extra']