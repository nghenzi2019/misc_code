# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:01:28 2019

@author: hh_s
"""

from opcua import ua, Server

import time
import numpy as np
import datetime
from random import choice

def server_opc():
    server = Server()
    server.set_endpoint("opc.tcp://0.0.0.0:4840/vm_opc")
    
    
    server.set_server_name("Servidor notificaciones vm")
    # setup our own namespace, not really necessary but should as spec
    uri = "notif VM"
    
    idx = server.register_namespace(uri)
    # get Objects node, this is where we should put our nodes
    objects = server.get_objects_node()
    
    # populating our address space
    myobj = objects.add_object(idx, "notif_vm")
    
    
    
    event_id=myobj.add_variable(idx, "ID evento", float(0))
#    new_st.set_writable()
    
    tss=myobj.add_variable(idx, "Timestamp", datetime.datetime.now())
#    new_st.set_writable()
    
    progresiva=myobj.add_variable(idx, "Progresiva", float(-1))
    
    estado_alarma=myobj.add_variable(idx, "Estado alarma", 'inicio')
    
    criticidad=myobj.add_variable(idx, "Criticidad", float(-1))
    
    ack_bit=myobj.add_variable(idx,"ACK bit",False)
    ack_bit.set_writable()
    
    alarma_activa=myobj.add_variable(idx,"Alarma presente",False)
    # starting!
    tags={'id':event_id,'tss':tss,'prog':progresiva,'st':estado_alarma,'crit':criticidad,'ack':ack_bit,'active':alarma_activa}
    server.start()
    try:
        while True:
            if np.random.uniform()>=0.5:
                tags['id'].set_value(float(np.random.randint(500)))
                tags['tss'].set_value(datetime.datetime.now())
                tags['st'].set_value(choice(['inicio', 'cambio', 'fin']))
                
            else:
                print 'Bit ACK: ', tags['ack'].get_value(), datetime.datetime.now()
                time.sleep(3)
    finally:
    #close connection, remove subcsriptions, etc
        server.stop()
