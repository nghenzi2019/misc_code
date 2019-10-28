import multiprocessing as mp
import threading as th
import matplotlib.pyplot as plt
import numpy as np
import time
import collections

def add_line(data):
	return data.shape,np.max(data)

def update_image(image,new_rows):
    prev_data=image.get_array()
    rows_to_update=new_rows.shape[0]
    new_data=np.append(prev_data[rows_to_update:,:],new_rows,axis=0)
    image.set_data(new_data)

def plot_binary(q,img,f):
	ii=0
	while True:
		if (q.qsize())>0:
			data,ts=q.get()
			print im.get_array()
			update_image(img,data>20)
			ii+=1
			if ii>5:
				f.canvas.draw()
				fig.canvas.set_window_title(ts)
				ii=0

def data_gen(q):
	for i in range(500):
		dd=np.random.uniform(high=i+1,size=(1,5))
		print dd
		q.put_nowait((dd,str(i)))
		time.sleep(0.3)


if __name__=="__main__":
	
	##ADD MAIN

	queue=mp.Queue()
	#p = mp.Process(target=plot_binary, args=(queue,))
	#p.daemon=True
	#p.start()

	##//ADD MAIN
	
	fig,ax=plt.subplots()
	im=ax.imshow(np.zeros((5,5),dtype=bool),vmax=5)
	
	p_gen=mp.Process(target=data_gen,args=(queue,))
	p_gen.daemon=True
	
	th_plot=th.Thread(target=plot_binary,args=[queue,im,fig])
	th_plot.setDaemon(True)
	
	th_plot.start()
	p_gen.start()
	
	plt.show()
	print "sleeping 5 secs"
	time.sleep(5)
