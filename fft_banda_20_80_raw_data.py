def filtro_banda(chunk,fila_freq_ini,fila_freq_fin):
	fft_banda=np.abs(np.fft.rfft(chunk,axis=0))
	return np.sum(fft_banda[fila_freq_ini:fila_freq_fin,:],axis=0)


freq_ini=20
freq_fin=80
dFreq=1

shots=matriz1.shape[0]
shots_por_chunk=2000

filtro_res=filtro_banda(matriz1[0:shots_por_chunk,:],freq_ini*dFreq,freq_fin*dFreq)

for i in range(1,shots/shots_por_chunk):
	filtro_res=np.vstack((filtro_res,filtro_banda(matriz1[i*shots_por_chunk:(i+1)*shots_por_chunk,:],freq_ini*dFreq,freq_fin*dFreq)))
