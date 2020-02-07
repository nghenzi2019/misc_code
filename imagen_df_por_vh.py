import itertools
from operator import itemgetter
import pickle
import numpy as np
import matplotlib.pyplot as plt

def agrupar_por_vh(list_dicc):
    res = sorted(list_dicc, key=itemgetter('vehiculo'))#ordeno primero porque groupby necesita ordenados para agrupar por item como necesitamos
    return itertools.groupby(res,key=itemgetter('vehiculo'))
    
def data_por_vh(df_files):
    
    max_t_por_vh={}
    cant_clusters_por_vh=0
    resumenes=[]
    
    for df_file in df_files:
        with open(df_file,'rb') as df:
            df_res=pickle.load(df)
            cluster=df_res['df']#agarro el dataframe

        vh=cluster['vehiculos'][0]
        
        
        if vh in cant_clusters_por_vh:
            cant_clusters_por_vh[vh]+=len(cluster.index)
        else:
            cant_clusters_por_vh[vh]=len(cluster.index)
        
        
        if vh in max_t_por_vh:
            if max(cluster['t'])>max_t_por_vh[vh]:
                max_t_por_vh[vh]=max(cluster['t'])
        else:
            max_t_por_vh[vh]=max(cluster['t'])
            
        resumenes.append(resumen_cluster(cluster))
        
               
    print max_t_por_vh, cant_clusters_por_vh
    return max_t_por_vh,cant_clusters_por_vh,agrupar_por_vh(resumenes)

def crear_matrices_max(clusters_cant_por_vh,max_t_por_vh):
    matrices_por_vh={}
    for vh,t in max_t_por_vh.iteritems():
        matrices_por_vh[vh]=np.empty((clusters_cant_por_vh[vh],max_t_por_vh[vh]))
        matrices_por_vh[vh].fill(np.nan)
        
    return matrices_por_vh


def llenar_matrices(matrices_por_vh,resumenes_agrupados):
    
    for vh, resumen in resumenes_agrupados:
        for i in range(matrices_por_vh[vh].shape[0]):
            resumen=resumen.next()#resumen es un iter,ahora es un dict
            fila_mod=matrices_por_vh[vh][i,:]
            
            desc=resumen['intervalo desconocido'] 
            pred_ok=resumen['posiciones acertadas']
            pred_notok=resumen['posiciones peligrosas']
            
            fila_mod[pred_ok]=1
            fila_mod[desc]=0
            fila_mod[pred_notok]=-1
            
            matrices_por_vh[vh][i,:]=fila_mod
            
import matplotlib
cmap_nan=matplotlib.cm.jet
cmap_nan.set_bad(color='gray',alpha=0.01)#color map para nan

#plt.imshow(....,cmap=cmap_nan)
