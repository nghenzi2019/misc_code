    
espectro=np.abs(np.fft.rfft(raw_cluster, axis = 0))
intMax=np.ones(espectro.shape[1])
pathRed='red_entrenada_concamiones.fit'
predict = prediceCircula(espectro, intMax, pathRed, 'normal')

def prediceCircula(inputFft, intMax, pathRed, predictForm = 'normal'):

    fileO = open(pathRed, 'rb')
    entrenamiento = pickle.load(fileO)
    fileO.close()

    N_bandas = 100
    ne=inputFft.shape[0]
    espectro=abs(inputFft)
    
    dFrec=2000./ne
    
    area_espectros={}
    ff=0
    
    E_features=np.zeros([N_bandas, espectro.shape[1]]) ## Hacemos una matriz, filas=Numero de bandas, columnas=Numero de espectros
    
    for k in range(espectro.shape[1]):  
        area_espectros[k]= np.cumsum(espectro[:,k])*dFrec #La suma acumulada de los vectores (funcion creciente)
        
        def f(x):
            return np.int(x)
    
        f2 = np.vectorize(f)
    
    ### tomo 100 bandas de frecuencia en escala lineal para el Bandpass
        
        BP_ini= np.arange(1,N_bandas+1)
        BP_ini = f2(BP_ini/ dFrec)
 
        BP_fin= np.arange(2,N_bandas+2)
        BP_fin = f2(BP_fin/ dFrec)
    
        Energy= area_espectros[ff][BP_fin]-area_espectros[ff][BP_ini]
        
        E_features[:,ff]=Energy
    
        ff+=1
        
    E_features_T=np.transpose(E_features)
    if predictForm == 'normal':
        predict = entrenamiento.predict(E_features_T)
         
        i = 0
        for i in range(len(predict)):
            try:
                if intMax[i] <= 0.04:
                    predict[i] = 'blanco'
            except:
                pass
    
            i += 1
    elif predictForm == 'probability':
        predict = entrenamiento.predict(E_features_T)
         
        i = 0
        for i in range(len(predict)):
            try:
                if intMax[i] <= 0.04:
                    predict[i][0] = 0.
                    predict[i][1] = 0.
                    predict[i][2] = 0.
                    predict[i][3] = 0.
            except:
                pass
    
            i += 1

    return predict
