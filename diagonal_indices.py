# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 00:30:41 2019

@author: hh_s
"""

import numpy as np
#%%
a=np.eye(5)

np.indices(2,3)

offset=
offset_array= #array como columna para desplazar todas si no empieza del (0,0)


matrix=[]
ventana_tam=40
fil_vent,col_vent=np.indices((ventana_tam,matrix.shape[1])) #genera los inidices de todas las filas y columnas de interes

fil_vent=fil_vent+np.arange(fil_vent.shape[1])#(muevo  la grilla sobre la diagonal)
fil_vent=fil_vent+offset_array#si no empieza en la fila 0, asume que hace etodas las columnas