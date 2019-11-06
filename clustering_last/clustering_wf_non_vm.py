import numpy as np
import matplotlib.pyplot as plt
import time
#from astropy.stats import median_absolute_deviation
import threading as th

from sklearn.cluster import DBSCAN
import matplotlib.patches as ptc
import pandas as pd

import warnings
warnings.simplefilter('ignore',np.RankWarning)

def cluster_box(cluster):

    lower_left=np.array([np.min(cluster[:,0]),np.min(cluster[:,1])])

    width=np.linalg.norm(lower_left-np.array([np.max(cluster[:,0]),lower_left[1]]))

    height=np.linalg.norm(lower_left-np.array([lower_left[0],np.max(cluster[:,1])]))



    return lower_left,width,height

def cluster_fit(db,pts):

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

    core_samples_mask[db.core_sample_indices_] = True

    labels = db.labels_

    unique_labels = set(labels)

    

    clusters=[]

    noise_pts=pts[labels == -1]

    

    for k in unique_labels:

        if k!=-1:

            class_member_mask = (labels == k)

            

            x=pts[class_member_mask & core_samples_mask]

            ajuste=np.poly1d(np.polyfit(x[:,0], x[:,1], 1))

#            print "pendiente ajuste cluster", k ,ajuste[1]

    #           ax.plot(np.unique(x[:,0]), ajuste(np.unique(x[:,0])))

            

            xy = pts[class_member_mask & core_samples_mask]

            cs_cnt=len(xy)

#            if k>=0:

#                print cs_cnt, ' core_samples'

            xy = pts[class_member_mask & ~core_samples_mask]

            not_cs_cnt=len(xy)

#            if k>=0:

#                print len(xy), ' aislados'

            clusters.append((pts[class_member_mask],cs_cnt,not_cs_cnt,ajuste))

    

    return clusters,noise_pts

def ts_to_stringDate(ts):
    res=""
    for num in map(str,ts[0:3]):
        if len(num)==1:
            res+="0"
        res+=num+"/"
    res=res[:-1]
    res+=" "
    for num in map(str,ts[3:]):
        if len(num)==1:
            res+="0"
        res+=num+":"
    return res[:-1]

def update_image(image,new_rows):
    prev_data=image.get_array()
    if len(new_rows.shape)>1:
        rows_to_update=new_rows.shape[0]
    else:
        rows_to_update=1
    new_data=np.append(prev_data[rows_to_update:,:],new_rows.reshape(rows_to_update,new_rows.size/rows_to_update),axis=0)
    image.set_data(new_data)

def agrupar_rango(array,dist):#greedy, no es lo mejor para encontrar rangos menores que dist pero 

    array.sort()
    array_diff = np.ediff1d(array)
    
    curr_cnt=0
    range_found=False
    sol=[]

    for i,a in enumerate(array[:-1]):
    	d=array_diff[i]
    
    	if d>dist:
    		if not range_found:
    			sol.append([a])
    		else:
    			curr_cnt=0
    			range_found=False
    			sol.append(interval)
    
    	else:
    		if not range_found:
    			range_found=True
    			interval=[array[i],array[i+1]]
    			curr_cnt+=d
    		else:
    			if curr_cnt+d<dist:
    				interval.append(array[i+1])
    				curr_cnt+=d 
    			else:
    				curr_cnt=0
    				range_found=False
    				sol.append(interval)
    #ultima iteracion para cerrar el ultimo item
    if not range_found:
    	sol.append([array[-1]])
    else:
    	sol.append(interval)
    
    return sol


def lineas_importantes(pts,cant=13,dist=100):#FALTA MERGE DE PROXIMOS EN Y y considerar manchas
    pts_df=pd.DataFrame(pts,columns=['x','y'])
    y_imp=[]
    agrupados_por_y=pts_df.groupby('y')
#    print agrupados_por_y.groups
    for y,XYs in agrupados_por_y:
        current_sum_y=XYs.x.size
        if y+1 in agrupados_por_y.indices:
            current_sum_y+=agrupados_por_y.get_group(y+1).x.size
        if y-1 in agrupados_por_y.indices:
            current_sum_y+=agrupados_por_y.get_group(y-1).x.size
        if current_sum_y>=cant:
            y_imp.append(y)
    
    res_y={}
    
    for y in y_imp:
        ys_cercanos=[y]
        
        if y+1 in y_imp:
            ys_cercanos.append(y+1)
            y_imp.remove(y+1)
        if y-1 in y_imp:
            ys_cercanos.append(y-1)
            y_imp.remove(y-1)
            
        test_zone=[]
        for yss in ys_cercanos:
            test_zone.extend(list(agrupados_por_y.get_group(yss).x))
        rangos=agrupar_rango(test_zone,dist)
        #print rangos,ys_cercanos
    
        for r in rangos:
            if len(r)>cant:
                res_y[y]=(min(r),max(r))
    
    return res_y


def keys_dist(d,y0,dist):
    res=0
    max_key=0
    x0min=d[y0][0]
    for k in d.keys():
        if abs(y0-k)<dist and ((d[k][0]-50< x0min <d[k][1]+50)) and abs( d[k][1] - d[k][0] ) > 50:#a distancia dist y los x cerca
            res+=1
            max_key=max(max_key,k)
    return res,max_key


def plot_binary(q,ax,img,fig,filas_imagen_binaria,mad_data,median_data,base_fin,bin_filt_1,eps,min_samples):
    
    wf_actual=np.zeros((filas_imagen_binaria,median_data.size))
    wf_actual=wf_actual>bin_filt_1
    
    
    wf_posible_exc=np.zeros((filas_imagen_binaria,median_data.size))
    wf_posible_exc=wf_posible_exc>80#80 es bin_filt_2
#    wf_actual=img.get_array()
#    print wf_actual[0:10,0:10]
    update_plot_after=10
    update_plot_after_cnt=0
    
    bordes=[]
    while True:
        if (q.qsize())>0:
            
            dd  =q.get()
            
            wf_actual=np.roll(wf_actual,-1,axis=0)
            MAD=np.abs(dd[:base_fin]-median_data)/mad_data
            binary_dd=MAD>bin_filt_1
            wf_actual[-1,:]=binary_dd

            wf_posible_exc=np.roll(wf_posible_exc,-1,axis=0)          
            wf_posible_exc[-1,:]=MAD>80
            

#            print binary_dd.shape
#            img.set_data(wf_actual)
            
            #update_image(img,np.ones((5,MAD.size),dtype=bool))
#            update_image(img,binary_dd)
            update_plot_after_cnt+=1            
            if update_plot_after_cnt == update_plot_after:
                update_plot_after_cnt=0
#                update_image(img,MAD.reshape(-1,MAD.size)>bin_filt_1)
            #Clustering
                pts_cluster=np.array(zip(np.nonzero(wf_actual)[1],np.nonzero(wf_actual)[0]))
                db=DBSCAN(eps=eps,min_samples=min_samples).fit(pts_cluster)       
                rr,noise_pts=cluster_fit(db,pts_cluster)
                
                for b in bordes:
                    b.set_visible(False)
                bordes=[]

                posibles_excav=[]
                for cluster,cs,not_cs,fit in rr:
                    ll,w,h=cluster_box(cluster)
                    
                    #if np.abs(fit[1])>0.4:#con inclinacion
                    if 0.75<=np.abs((h/w)/fit[1])<=1.4:
                        edgecolor='g'
                        txt_c='r'
                        fontsize=22
                    else:
                        edgecolor='r'
                        txt_c='tab:gray'
                        fontsize=18
                        
                    if h>60:
                            
                        rect_clust=ptc.Rectangle(ll,w,h,facecolor='none',edgecolor=edgecolor)
                        ax.add_patch(rect_clust)
                        bordes.append(rect_clust)
                        tr=(ll[0]+w,ll[1]+h)
                        txt_data=' '.join(['x:',str(int(w)),'y:',str(int(h)),'\nLL:',str(ll),'\n km/h:',str(4.06*3.6/round(fit[1],3)),'\n relacion pendiente-caja:',str(round((h/w)/fit[1],3))])
                        if edgecolor=='r' or edgecolor=='g':
                            ann=ax.annotate(txt_data,tr,color=txt_c,fontsize=fontsize)
                            bordes.append(ann)
                    if h<=80 and w<100 and 70<ll[1]<300: #agrego los puntos de clusters
                        for x in cluster:
                            posibles_excav.append(x)
                        
                        
                    #print ts_to_stringDate(ts), ll,w,h
                
#                wf_filt=wf_mad[:,bin_inicio:]>80
                intensos=np.array(zip(np.nonzero(wf_posible_exc)[1],np.nonzero(wf_posible_exc)[0]))
                
                ruidosos=set(tuple(x) for x in noise_pts)
                posibles_excav=set(tuple(x) for x in posibles_excav)
                
                posibles_excav=posibles_excav.union(ruidosos)
                
                ruidosos_intensos=np.array([x for x in set(tuple(x) for x in intensos) & set(tuple(x) for x in posibles_excav)])
#                ruidosos_intensos=np.array([x for x in set(tuple(x) for x in noise_pts)])
                #rectas con mas de 10
                if ruidosos_intensos.size>0:
                    
                    y_imp=lineas_importantes(ruidosos_intensos)
                    for y,(xmin,xmax) in y_imp.items():
                        if not( 1500<xmin<2200 or 1700<xmax<2400):#saco circunvalacion
                            
                            lineas_cercanas, y_mas_alejada=keys_dist(y_imp,y,80)
                            if lineas_cercanas<4:
                                
                                linea_exca=ax.axhline(y=y,color='r')
                                bordes.append(linea_exca)
                                ssca_noise=ax.scatter([xmin,xmax],[y,y],s=200*np.ones(ruidosos_intensos.shape),color='black')
                                bordes.append(ssca_noise)
                            else:
                                print y_imp.keys(), lineas_cercanas,y
                                
                                if abs(y-y_mas_alejada)>10: #para no dibujar rects muy chicos
                                    rect_clust=ptc.Rectangle((xmin,y),abs(xmax-xmin),abs(y-y_mas_alejada),facecolor='none',edgecolor='blue')
                                    ax.add_patch(rect_clust)
                                    bordes.append(rect_clust)
                                    ann=ax.annotate('EXCAVACION POSIBLE',(xmax,y_mas_alejada),color='blue',fontsize=25)
                                    bordes.append(ann)
                
                update_image(img,wf_actual[-update_plot_after:,:])
                #fig.canvas.set_window_title('WF BINARIO cluster: '+ts_to_stringDate(ts))
                fig.canvas.draw()            
        else:
            time.sleep(0.1)


def crit_clust(w,l,ll,vel):
    pass


def cluster_proc(q,mad_data,median_data):
    base_fin=median_data.size


    wf_base=[]#less ram maybe
    
    print "mediana cluster calculada"
    
    bin_filt_1=25#parametro cluster primera filtrada
    eps=6#7
    min_samples=56#80
    
    fig,ax=plt.subplots()

    filas_imagen_binaria=400
    size_wf=(filas_imagen_binaria,median_data.size)

    im=ax.imshow(np.eye(N=filas_imagen_binaria,M=median_data.size)>0.5,aspect='auto',cmap='binary')
#    im=ax.imshow(np.zeros((size_wf)),aspect='auto', cmap='inferno',vmax=3)
        
       
    th_plot=th.Thread(target=plot_binary,args=[q,ax,im,fig,filas_imagen_binaria,mad_data,median_data,base_fin,bin_filt_1,eps,min_samples])
    th_plot.setDaemon(True)
    th_plot.start()
    
    plt.show()
    
