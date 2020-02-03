import numpy as np
from collections import Counter

def predict_cluster(spd,w_x,i_x,t_bin):#spd en km/h

	retro=(spd<10) and (t_bin>160)
	camioneta= any( [(spd>20) and (t_bin>50) , (w_x < spd+2) and (10<spd<20) ] )
	camion= (14<spd<20) and (w_x>=spd+2) 
	sesenta_cuarenta= (10<spd<=14) and(w_x>=spd+2)

	if retro: return 'retro'
	if camioneta: return 'camioneta'
	if camion: return 'camion'
	if sesenta_cuarenta: return'60 C 40 R'


def resumen_cluster(df): #calcula estadisticas sobre el cluster

	errores_posibles={'camioneta':-1,'camion':-2,'retro':-3}

	predicciones=np.array(df['predict'])
	vehiculos=np.array(df['vehiculo'])
	tiempos=np.array(df['t'])
	codigo_error=map(errores_posibles.get,predicciones)

	predict_result=np.where(predicciones==vehiculos ,tiempos,codigo_error) 
	#donde acierto guardo df['t'], donde no guardo el codigo de errores_posibles o None si es ovni la prediccion

	print predict_result

	posiciones_acertadas=np.where(predict_result[0]>=0)
	

	aciertos_totales=posiciones_acertadas[0].size

	tiempos_acertados=np.unique(predict_result[posiciones_acertadas]) #saco duplicados de los t acertados
	errores_peligrosos=np.where(predict_result[0]==errores_posibles['retro'])[0].size




def resumen_por_vh(dffs,vhs): #devuelve diccionario con aciertos por vehiculo

	predict_ok={}
	for vh in set(vhs):
		predict_ok[vh]=0#acertados


	for df,vh in zip(dffs,vhs):
		
		if df['prediccion']==vh:
			predict_ok[vh]+=1

	freq_vh=Counter(vhs)
	for vh in set(vhs):
		predict_ok[vh]/=freq_vh[vh]#divido por las apariciones

	print predict_ok

	return predict_ok
