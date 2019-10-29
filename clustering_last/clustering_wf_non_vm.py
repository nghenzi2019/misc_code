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


def plot_binary(q,ax,img,fig,filas_imagen_binaria,mad_data,median_data,base_fin,bin_filt_1,eps,min_samples):
    
    wf_actual=np.zeros((filas_imagen_binaria,median_data.size))
    wf_actual=wf_actual>bin_filt_1
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


                for cluster,cs,not_cs,fit in rr:
                    ll,w,h=cluster_box(cluster)
                    
                    #if np.abs(fit[1])>0.4:#con inclinacion
                    if 0.5<=np.abs((h/w)/fit[1])<=2:
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
                        txt_data=' '.join(['x:',str(int(w)),'y:',str(int(h)),'\npts:',str(int(len(cluster))),'\n pendiente:',str(round(fit[1],3)),'\nrelacion pendiente caja:',str(round((h/w)/fit[1],3))])
                        if edgecolor=='r' or edgecolor=='g':
                            ann=ax.annotate(txt_data,tr,color=txt_c,fontsize=fontsize)
                            bordes.append(ann)
                    #print ts_to_stringDate(ts), ll,w,h
                      
                update_image(img,wf_actual[-update_plot_after:,:])
                #fig.canvas.set_window_title('WF BINARIO cluster: '+ts_to_stringDate(ts))
                fig.canvas.draw()            
        else:
            time.sleep(0.1)



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
    
