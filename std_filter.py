import numpy as np
import matplotlib.pyplot as plt
import glob,os,time

def mv_avg_std(matrix,window,window_z):

    res=np.zeros((matrix.shape[0]-window+1,matrix.shape[1]))#puede salir si mejora tiempos
    res[0]=np.mean(matrix[:window,:],axis=0)    
    suma_actual=np.sum(matrix[:window,:],axis=0)

    for i in range(1,matrix.shape[0]-window+1):
        suma_actual=suma_actual-matrix[i-1,:]+matrix[i+window-1,:]
        res[i]=suma_actual/window

    if window_z>1:
        res[:,0]=np.mean(res[:,:window_z],axis=1)
        suma_actual_z=np.sum(res[:,:window_z],axis=1)

        for j in range(1,matrix.shape[1]-window_z+1):
            suma_actual_z=suma_actual_z-res[:,j-1]+res[:,j+window_z-1]
            res[:,j]=suma_actual_z/window_z    

    return np.std(res[:,:matrix.shape[1]-window_z+1],axis=0), np.mean(matrix[:,:matrix.shape[1]-window_z+1],axis=0)

def std_filter(raw_m,shotPorChunk=1000,window_time=40,window_z=1):
    n=raw_m.shape[0]/shotPorChunk
    wf=np.zeros((n,raw_m.shape[1]-window_z+1))
    wf_avg=np.zeros((n,raw_m.shape[1]-window_z+1))
    for i in xrange(n):
        esteChunk=raw_m[i*shotPorChunk:(i+1)*shotPorChunk,:] 
        esteChunkSTD,esteChunkAVG=mv_avg_std(esteChunk,window_time,window_z)
        wf[i,:]=esteChunkSTD
        wf_avg[i,:]=esteChunkAVG
    return wf,wf_avg

def moving_average(a, n=300) :
    ret = np.cumsum(a, dtype=float,axis=1)
    ret[:,n:] = ret[:,n:] - ret[:,:-n]
    return ret[:,n - 1:] / n


#%%
for archivo in glob.glob('*.bin'):
    bins_raw=2000
    print "filtrando ",archivo
    st=time.time()
    data=np.fromfile(archivo,dtype=np.int16).reshape(-1,bins_raw)
    wf,wf_avg=std_filter(data)
    moving_average(wf).astype(np.float32).flatten().tofile(archivo[:-3]+'std')
    print "creado ", archivo[:-3]+'std'
    moving_average(wf_avg).astype(np.float32).flatten().tofile(archivo[:-3]+'avg')
    print "creado ", archivo[:-3]+'avg'
    print "tarda ", (time.time()-st), "seg por archivo\n"
