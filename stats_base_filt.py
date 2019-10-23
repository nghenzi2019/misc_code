# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 22:51:42 2019

@author: hh_s
"""

"""
Created on Sat Sep 21 22:58:12 2019

@author: hh_s
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import matplotlib.patches as ptc
#from sklearn.neighbors import NearestNeighbors
from astropy.stats import median_absolute_deviation
import pandas as pd



base_Std=np.load("../bases comprimidas/stds_Base.npy")
base_avg=np.load("../bases comprimidas/avg_Base.npy")
wf_base=base_Std/base_avg

#%%lets go MAD
mad_data=median_absolute_deviation(wf_base,axis=0)
median_data=np.median(wf_base,axis=0)
#%% DBSCAN
def plot_clusters(db,pts):
        
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
    fig,ax=plt.subplots()
    ax.invert_yaxis()
    for k, col in zip(unique_labels, colors):
        if k == -1:
        # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
        
        x=pts[class_member_mask & core_samples_mask]
        
        if x.size>0 and k!=-1:
        	print "pendiente ajuste cluster", k ,np.poly1d(np.polyfit(x[:,0], x[:,1], 1))[1]
        	ax.plot(np.unique(x[:,0]), np.poly1d(np.polyfit(x[:,0], x[:,1], 1))(np.unique(x[:,0])))
        
        xy = pts[class_member_mask & core_samples_mask]
        if k>=0:
            print len(xy), ' core_samples'
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=18)
        
        xy = pts[class_member_mask & ~core_samples_mask]
        if k>=0:
            print len(xy), ' aislados'
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=2)
        
        ax.set_title('Estimated number of clusters: %d' % n_clusters_)

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
            print "pendiente ajuste cluster", k ,ajuste[1]
    #        	ax.plot(np.unique(x[:,0]), ajuste(np.unique(x[:,0])))
            
            xy = pts[class_member_mask & core_samples_mask]
            cs_cnt=len(xy)
            if k>=0:
                print cs_cnt, ' core_samples'
            xy = pts[class_member_mask & ~core_samples_mask]
            not_cs_cnt=len(xy)
            if k>=0:
                print len(xy), ' aislados'
            clusters.append((pts[class_member_mask],cs_cnt,not_cs_cnt,ajuste))
    
    return clusters,noise_pts

def lineas_importantes(pts,cant=10):
    pts_df=pd.DataFrame(pts,columns=['x','y'])
    res_y=[]
    agrupados_por_y=pts_df.groupby('y').groups
    for k,v in agrupados_por_y.items():
        current_sum_k=v.size
        if k+1 in agrupados_por_y.keys():
            current_sum_k+=agrupados_por_y[k+1].size
        if k-1 in agrupados_por_y.keys():
            current_sum_k+=agrupados_por_y[k-1].size
        if current_sum_k>=cant:
            res_y.append(k)
    return set(res_y)
#%%
wf_real=np.load('0407_2.npy')
filas_por_figura=1000
bin_inicio=1000
n=wf_real.shape[0]/filas_por_figura
for i in range(11,12):
    
#    fig=plt.figure(figsize=(12, 12))
    wf_actual=wf_real[i*filas_por_figura:(i+1)*filas_por_figura,:]
    wf_mad=np.abs(wf_actual-median_data)/mad_data
#    fig.add_subplot(1,2,1)
#    plt.imshow(wf_mad>50,aspect='auto',cmap=plt.cm.gray)
#    continue
#    wf_act=wf_real[i*1000:(i+1)*1000,:]
#    plt.imsave(str(i)+'.png',wf_act[:,1000:6200])
    wf_filt=wf_mad[:,bin_inicio:]>30
#    wf_filt=wf_actual[:,bin_inicio:]>0.04
    pts=np.array(zip(np.nonzero(wf_filt)[1],np.nonzero(wf_filt)[0]))
    
#    vecindad=NearestNeighbors(n_neighbors=400)
#    vecinos=vecindad.fit(pts)
#    dst,idx=vecinos.kneighbors(pts)
#    dist=np.sort(dst,axis=0)
#    dist=dist[:,1]
#    fig2,ax2=plt.subplots(figsize=(10,10))
#    ax2.plot(dist)
#    ax2.set_ylim([0,100])

#    for k in [60]:
#        plt.imsave(str(i)+'_'+str(k)+'.png',wf_mad[:,1000:6200]>k)
#    fig.add_subplot(1,2,2)
    
    fig,ax=plt.subplots(figsize=(12,12))

#    db=DBSCAN(eps=25,min_samples=400).fit(pts)
    db=DBSCAN(eps=15,min_samples=250).fit(pts)

    rr,noise_pts=cluster_fit(db,pts)

    ax.imshow(wf_real[i*filas_por_figura:(i+1)*filas_por_figura,bin_inicio:],aspect='auto',vmax=0.2,cmap='jet')
    for cluster,cs,not_cs,fit in rr:
        ll,w,h=cluster_box(cluster)
        if np.abs(fit[1])>0.4:#con inclinacion
            edgecolor='r'
            txt_c='r'
            plot_fast=True
        else:
            edgecolor='g'
            txt_c='w'
            plot_fast=True
        if plot_fast:
            ax.add_patch(ptc.Rectangle(ll,w,h,facecolor='none',edgecolor=edgecolor))
            tr=(ll[0]+w,ll[1]+h)
            roi=wf_mad[int(ll[1]):int(tr[1]),bin_inicio+int(ll[0]):bin_inicio+int(tr[0])]
    #        criterio:(float(cs)/(len(cluster))) ////// len(np.where(roi>80)[0])
            integral_idea=np.sum(roi[np.where(roi>80)])/(w*h)
            txt_data='sum: '+str(round(integral_idea,4))
            txt_data+='\nPendiente ajuste: '+str(round(fit[1],4))
            txt_data+='\nproporcion cluster: '+str(len(cluster)/(w*h))
            ax.annotate(txt_data,tr,color=txt_c)
#        ax.plot(noise_pts[1],noise_pts[0],'.')
    
    #2do filtrado
    wf_filt=wf_mad[:,bin_inicio:]>100
    pts2=np.array(zip(np.nonzero(wf_filt)[1],np.nonzero(wf_filt)[0]))
    
    ruidosos_intensos=np.array([x for x in set(tuple(x) for x in pts2) & set(tuple(x) for x in noise_pts)])
    
    #rectas con mas de 10
    y_imp=lineas_importantes(ruidosos_intensos)
    for l in y_imp:
        ax.axhline(y=l,color='r')
#    unique, counts = np.unique(ruidosos_intensos[:,1], return_counts=True)
#    cuentas_por_y=np.asarray((unique, counts)).T
#    for i in range(1,unique_counts.size-1):#voy desde el 2do hasta el anteultimo
#        cuenta_local=cuentas_por_y[i-1]+
    ax.scatter(ruidosos_intensos[:,0],ruidosos_intensos[:,1],s=200*np.ones(ruidosos_intensos.shape),color='black')
#    db_ri=DBSCAN(eps=15,min_samples=60).fit(ruidosos_intensos)
#    rr2,_=cluster_fit(db_ri,ruidosos_intensos)
#
#    for cluster,cs,not_cs,fit in rr2:
#        ll,w,h=cluster_box(cluster)
#        if np.abs(fit[1])>0.4:#con inclinacion
#            edgecolor='y'
#            txt_c='y'
#        else:
#            edgecolor='y'
#            txt_c='y'
#        ax.add_patch(ptc.Rectangle(ll,w,h,facecolor='none',edgecolor=edgecolor))
#        tr=(ll[0]+w,ll[1]+h)
#        roi=wf_mad[int(ll[1]):int(tr[1]),bin_inicio+int(ll[0]):bin_inicio+int(tr[0])]
#    #        criterio:(float(cs)/(len(cluster))) ////// len(np.where(roi>80)[0])
#        integral_idea=np.sum(roi[np.where(roi>80)])/(w*h)
#        txt_data='sum: '+str(round(integral_idea,4))
#        txt_data+='\nPendiente ajuste: '+str(round(fit[1],4))
#        txt_data+='\nproporcion cluster: '+str(len(cluster)/(w*h))
#        ax.annotate(txt_data,tr,color=txt_c)


