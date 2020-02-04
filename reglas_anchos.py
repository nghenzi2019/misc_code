import numpy as np
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

	

	return resultado

##Ejemplo plot prediccion(t)
dd=resumen_cluster(df22) #dd es un diccionario con los resultados de resumen_cluster

desc=np.zeros_like(dd['intervalo desconocido']) #grafico como 0 los que todavía no clasificó
pred_ok=np.zeros_like(dd['posiciones acertadas'])+1 #grafico como 1 las predicciones correctas
pred_notok=np.zeros_like(dd['posiciones peligrosas'])-1 #grafico como -1 los errores peligrosos (confusiones de retro)

plt.plot(dd['posiciones peligrosas'],pred_notok, '.',color='black',label=dd['errores peligrosos']*100)
plt.plot(dd['posiciones acertadas'],pred_ok, '.',color='blue',label=dd['aciertos']*100 )
plt.plot(dd['intervalo desconocido'],desc, '.',color='red',label=dd['desconocidos']*100)
plt.legend()
plt.show()
#FIN EJEMPLO

def resumen_por_vh(dffs,vhs): #devuelve diccionario con aciertos por vehiculo
#Todavia no sirve esta funcion
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
