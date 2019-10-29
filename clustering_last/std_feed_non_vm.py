import numpy as np
import multiprocessing as mp
import glob,os
import time
from astropy.stats import median_absolute_deviation

from clustering_wf_non_vm import cluster_proc


def std_feed_from_std_avg(q,std_files,avg_files,bins=22201):
	
	for f_std,f_avg in zip(std_files,avg_files):
		current_std=np.fromfile(f_std,dtype=np.float32).reshape(-1,bins)
		current_avg=np.fromfile(f_avg,dtype=np.float32).reshape(-1,bins)
		current_wf=current_std/current_avg
		for row in current_wf:
			q.put_nowait(row)
			time.sleep(0.01)

def std_feed_from_npy_std_avg(q,npy_std,npy_avg):

	current_std=np.load(npy_std)
	current_avg=np.load(npy_avg)
	current_wf=current_std/current_avg
	for row in current_wf:
		q.put_nowait(row)
		time.sleep(0.01)

def std_feed_from_npy_wf(q,npy_wf):

	current_wf=np.load(npy_wf)
	i=0
	for row in current_wf:
		i+=1
		q.put_nowait(row)
		print i, current_wf.shape[0]
		time.sleep(0.01)



std_files=glob.glob('0*.std')
avg_files=glob.glob('../AVG/0*.avg')

q=mp.Queue()

print "cargando base..."
   
base_Std=np.load("../bases comprimidas/stds_Base.npy")
base_avg=np.load("../bases comprimidas/avg_Base.npy")
wf_base=base_Std/base_avg

   

#base_inicio=1000
print "base cluster cargada"
mad_data=median_absolute_deviation(wf_base,axis=0)
median_data=np.median(wf_base,axis=0)

args=[q,mad_data,median_data]
if len(std_files)>0:
	clust_proc=mp.Process(target=cluster_proc,args=args)
	clust_proc.daemon=True
	clust_proc.start()
	std_feed_from_std_avg(q,std_files,avg_files)
	clust_proc.join()
else:
	npy_files=glob.glob('*.npy')
	for npy_f in npy_files:
		print npy_f
		clust_proc=mp.Process(target=cluster_proc,args=args)
		clust_proc.daemon=True
		clust_proc.start()
		std_feed_from_npy_wf(q,npy_f)
		clust_proc.join()
	
