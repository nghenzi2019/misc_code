import itertools
from operator import itemgetter
import pickle
import numpy as np
import matplotlib.pyplot as plt


from collections import Counter

def predict_cluster(spd,w_x,i_x,t_bin):#spd en km/h

    retro=(spd<10) and (t_bin>=160)
    camioneta= any( [(spd>20) and (t_bin>50) , (w_x < spd+2) and (10<spd<20) ] )
    camion= (14<spd<20) and (w_x>=spd+2) 
    sesenta_cuarenta= (10<spd<=14) and(w_x>=spd+2)

    if retro: return 'retro'
    if camion: return 'camion'
    if camioneta: return 'camioneta'
    if sesenta_cuarenta: return'60 C 40 R'

    #return 'ukw' #DEVUELVE NONE CUANDO NO SABE


def resumen_cluster(df): #calcula estadisticas sobre el cluster.

##Devuelve un diccionario con el vehiculo y los valores de aciertos, errores peligrosos y posiciones desconocidas

    errores_posibles={'camioneta':-1,'camion':-2,'retro':-3, '60 C 40 R':-3} #mismo codigo para retro y 60C40R

    predicciones=np.array(df['predict'])
    vehiculos=np.array(df['vehiculo'])
    
    if vehiculos[0]=='retro': #considero aciertos los 60/40 para retro y para camion. DISCUTIR
        predicciones[predicciones=='60 C 40 R']='retro'
    if vehiculos[0]=='camion':
        predicciones[predicciones=='60 C 40 R']='camion'
            
    tiempos=np.array(df['t'])
    codigo_error=map(errores_posibles.get,predicciones)

    predict_result=np.where(predicciones==vehiculos ,tiempos,codigo_error) 
    #donde acierto guardo df['t'], donde no guardo el codigo de errores_posibles o None si es ovni la prediccion

    #print predict_result

    posiciones_acertadas=np.where(predict_result>=0)

    aciertos_totales=posiciones_acertadas[0].size

    tiempos_acertados=np.unique(predict_result[posiciones_acertadas]) #saco duplicados de los t acertados
    
    if vehiculos[0]=='retro':
        posiciones_erradas_peligrosas=np.where((predict_result!=errores_posibles['retro']) & (predict_result!=None) & (predict_result<0) )

    else:
        posiciones_erradas_peligrosas=np.where(predict_result==errores_posibles['retro'])

    errores_peligrosos=posiciones_erradas_peligrosas[0].size
    intervalo_incertidumbre=np.where(predict_result==None)[0] #posiciones donde aun no se sabe que es

    subclusters_totales=float(len(df.index))

    
    resultado={'vehiculo':vehiculos[0]}

    resultado['aciertos']=(aciertos_totales)/(subclusters_totales)
    resultado['posiciones acertadas']=tiempos_acertados

    resultado['errores peligrosos']=errores_peligrosos/subclusters_totales
    resultado['posiciones peligrosas']=np.unique(tiempos[posiciones_erradas_peligrosas])

    resultado['intervalo desconocido']=np.unique(tiempos[intervalo_incertidumbre])
    resultado['desconocidos']=intervalo_incertidumbre.size/subclusters_totales
    
    resultado['max_t']=max(df['t'])

    

    return resultado



def agrupar_por_vh(list_dicc):
    res = sorted(list_dicc, key=itemgetter('vehiculo'))#ordeno primero porque groupby necesita ordenados para agrupar por item como necesitamos
    return itertools.groupby(res,key=itemgetter('vehiculo'))
    
def data_por_vh(df_files):
    
    max_t_por_vh={}
    cant_clusters_por_vh={}
    resumenes=[]
    
    cant_ignore=0
    
    for df_file in df_files:
        with open(df_file,'rb') as df:
            df_res=pickle.load(df)
            cluster=df_res['df']#agarro el dataframe

        vh=cluster['vehiculo'][0]
        
        
        if vh not in ['retro','camion','camioneta']:
            cant_ignore+=1
           # print vh,cant_ignore
            continue
        #if vh=='camioneta':
            #print resumen_cluster(cluster)
            #raw_input()
        
        
        if vh in cant_clusters_por_vh:
            cant_clusters_por_vh[vh]+=1#len(cluster.index)
        else:
            cant_clusters_por_vh[vh]=1#len(cluster.index)
        
        
        if vh in max_t_por_vh:
            if max(cluster['t'])>max_t_por_vh[vh]:
                max_t_por_vh[vh]=max(cluster['t'])
        else:
            max_t_por_vh[vh]=max(cluster['t'])
            
        resumenes.append(resumen_cluster(cluster))
        #resumenes.append(df_res['resumen'])
               
    print max_t_por_vh, cant_clusters_por_vh
    return max_t_por_vh,cant_clusters_por_vh,agrupar_por_vh(resumenes),resumenes

def crear_matrices_max(clusters_cant_por_vh,max_t_por_vh):
    matrices_por_vh={}
    for vh,t in max_t_por_vh.iteritems():
        matrices_por_vh[vh]=np.empty((clusters_cant_por_vh[vh],1+max_t_por_vh[vh]/10))
        matrices_por_vh[vh].fill(np.nan)
        
    return matrices_por_vh


def llenar_matrices(matrices_por_vh,resumenes_agrupados):#llena matrix y consume resumenes
    
    for vh, res_df in resumenes_agrupados:
        for i in range(matrices_por_vh[vh].shape[0]):
            resumen=res_df.next()#resumen es un dict
            fila_mod=matrices_por_vh[vh][i,:]
            
            desc=resumen['intervalo desconocido'].astype(np.int)/10
            pred_ok=resumen['posiciones acertadas'].astype(np.int)/10
            pred_notok=resumen['posiciones peligrosas'].astype(np.int)/10
            
            fila_mod[pred_ok]=1
            fila_mod[desc]=0
            fila_mod[pred_notok]=-1
            
            fila_mod[int(resumen['max_t']/10)]=2
            
            matrices_por_vh[vh][i,:]=fila_mod
            
import matplotlib
cmap_nan=matplotlib.cm.Accent
cmap_nan.set_bad(color='white',alpha=0.01)#color map para nan

#plt.imshow(....,cmap=cmap_nan)

import glob
df_res_files=glob.glob('*loop1*resumen*')



##### propiedades de contours -----------------------------------------


from skimage import measure

contours = measure.find_contours(wf_data, threshold) 

#%%%% 

dx,intX, d_tiempo,intT=witx(contour, wf_cut1)

def witx(c2,wf_cut):

#plt.plot(c2[:,1],c2[:,0],'o')
    
    delta_x=[]
    i_x=[]
    for i in range(int(min(c2[:,0]))+5, int(max(c2[:,0])), 10): ## vario tiempo , calculo delta x 
#        print i 
        x=c2[:,1]
        f=c2[:,0]
        g= np.array([i for _ in range(f.shape[0])])
        idx = np.argwhere(np.diff(np.sign(f - g)))
    
        ddx=  max(x[idx]) - min(x[idx])
        i_x.append( np.sum( wf_cut[i, int(min(x[idx])) : int(max(x[idx])) ] ) ) 
        delta_x.append(ddx)
    delta_posicion=np.mean(delta_x)
    intensidad_posicion=np.mean(i_x)

    delta_t=[]
    i_t=[]
    for i in range(int(min(c2[:,1]))+5, int(max(c2[:,1])), 10): ## vario posicion , calculo delta t 
#        print i 
        t=c2[:,0]
        f=c2[:,1]
        g= np.array([i for _ in range(f.shape[0])])
        idx = np.argwhere(np.diff(np.sign(f - g)))
    
        ddt=  max(t[idx]) - min(t[idx])
        i_t.append( np.sum( wf_cut[int(min(t[idx])) : int(max(t[idx])), i ] ) ) 
        delta_t.append(ddt)
    delta_tiempo=np.mean(delta_t)
    intensidad_tiempo=np.mean(i_t)
    
    return delta_posicion, intensidad_posicion, delta_tiempo, intensidad_tiempo
