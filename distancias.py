import math

def distancia(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def my_cluster(puntos,radio):
    coords=set(puntos)#conjunto de puntos
    C=[]#C es la lista de clusters
    while len(coords):#hasta que quede vacio
        x0=coords.pop()#saco uno
        cluster = [x for x in coords if distancia(x0,x)<= radio]#me quedo con los cercanos
        C.append(cluster+[x0])#agrego el inicial
        for x in cluster:#elimino los puntos del cluster
            coords.remove(x)
    return C
