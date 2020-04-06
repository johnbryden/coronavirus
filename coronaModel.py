import numpy as np

def model (vars,t,beta1,beta2,tau2,sigma):
    if t>=tau2:
        beta = beta2
    else:
        beta = beta1

#    print (t,tau2,beta)
    y = vars[0]
    z = vars[1]
    dydt = beta*y*(1-z) - sigma*y
    dzdt = beta*y*(1-z)
    dvarsdt = [dydt,dzdt]
    return dvarsdt


