import numpy as np
import matplotlib.pyplot as plt
import time
from astropy import median_absolute_deviation
import threading as th

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
    rows_to_update=new_rows.shape[0]
    new_data=np.append(prev_data[rows_to_update:,:],new_rows,axis=0)
    image.set_data(new_data)


def plot_binary(q,img,fig,mad_data,median_data,bin_filt_1):
	
	filas_imagen_binaria=400
	wf_actual=np.zeros((filas_imagen_binaria,median_data.size))
	
		
	
	while True:
		if (q.qsize())>0:
			dd , ts =q.get()
			
			wf_actual=np.roll(wf_actual,-1,axis=0)
			wf_actual[-1,:]=dd

			MAD=np.abs(wf_actual-median_data)/mad_data
			
			update_image(img,MAD>bin_filt_1)
			fig.canvas.draw()
		else:
			time.sleep(0.1)



def cluster_proc(q,size_wf):
	
	wf_base=
	mad_data=median_absolute_deviation(wf_base,axis=0)
	median_data=np.median(wf_base,axis=0)
	
	bin_filt_1=30#parametro cluster primera filtrada
	
	fig,ax=plt.subplots()
	im=ax.imshow(np.zeros((size_wf),dtype=bool))
	
	th_plot=th.Thread(target=plot_binary,args=[queue,im,fig,mad_data,median_data,bin_filt_1])
	th_plot.setDaemon(True)
	th_plot.start()
	
	plt.show()

