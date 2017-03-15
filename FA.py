import math
import matplotlib.pyplot as plt
import numpy as np


para = np.array([20, 500, 0.5, 0.2, 1])

n=para[0]  
MaxGeneration=para[1]
alpha=para[2]
betamin=para[3]
gamma=para[4]


# Numero total de Evaluaciones
NumEval=n*MaxGeneration

# Simple bounds/limits for d-dimensional problems
d = 20
Lb = np.zeros(d)
Ub = np.ones(d)

#Initial random guess
#u0 = Lb + (Ub-Lb) * random.rand(d)
#u0 = np.random.uniform(0,1,(20,d)) *(Ub-Lb)+Lb
#u1 = np.random.uniform(0,1,(20,d)) *(Ub-Lb)+Lb
#[u,eval,numeval] = ffa_mincon(@cost, u0, Lb, Ub, para)


zn=np.ones(n)*10**100



def alpha_new(alpha, NGen):
    delta=1-(10**(-4)/0.9)**(1/NGen);
    alpha=(1-delta)*alpha
    return alpha

def zeta(x, y):
    z= 20 + ((x**2)-10*math.cos(2*math.pi*x)) + ((y**2)-10*math.cos(2*math.pi*y))
    return z

#def unpack(p, pp):
#    lista_u0 = p
#    lista_u1 = pp
#    return map(zeta, lista_u0, lista_u1) 

def rangen():
    u0 = np.random.uniform(0,1,(d)) *(Ub-Lb)+Lb
    u1 = np.random.uniform(0,1,(d)) *(Ub-Lb)+Lb
    return (u0, u1) 

# parameters [n N_iteration alpha betamin gamma]
r = []
#MaxG = int(MaxGeneration)
for k in range(int(MaxGeneration)):
    alpha = alpha_new(alpha, MaxGeneration)
    tupla = rangen()
    results = map(zeta, tupla[0], tupla[1])
    results.sort()
    r.append(results[0])

r.sort()
print r[0]
    


#for i in range(d):
#    results = map(unpack, u0, u1)
#    r = results[i]
#    r.sort()
#    print r[0]
#    

#
##Display results
#bestsolution = u
#bestojb = eval
#total_numer_of_function_evaluations = numeval
#
##Cost or Objective function
#
#
#
######### Start FA ########
#
#def (nbest, fbest, NumEval = ffa_mincon(fhandle, u0, Lb, Ub, para)):
#    return None
#
##Check input  parameters (otherwise set as default values)
#
#def myplot(x,y, para = [20, 500, 0.25, 0.20, 1], Ub=[], Lb=[]):
#    print (Usuage FA_mincon(@cost, u0, Lb, Ub, para))
#    
#    
#n = para(1)
#MaxGeneration = para(2)
#alpha = (3)
#betamin = (4)
#gamma = (5)
#
#
## Total number of function evaluations
#NumEval = n*MaxGeneration
#
##Check if the upper bound & lower bound are the same size
#    if size(Lb) != size(Ub):
#         print("No son iguales Lb y Ub")
#         return
#         end 
#    
##Calculate dimension
#d = size(u0)
#
##Initial  values of an array
#zn = ones(n)*10**100
#
#
##generating the initial locations of n fireflies    
#[ns,Lightn] = init_ffa(n,d,lb,Ub,u0)    
#
#
##Iterations or pseudo time marching 
#
#for k in range (0,MaxGeneration):   #start iterations
#    
#    #This line of reducing alpha is optional
#    alpha = alpha_new(alpha,MaxGeneration);
#    
#    #Evaluate new solutions (for all n fireflies)
#    for i in range(0,n):
#        zn[i] = fhandle(ns[i:])
#        Lightn[i] = zn[i]
#
#    # ranking fireflies by their light intensity/Objectives
#    [Lightn, Index] = msort(zn) 
#    ns_tmp =ns
#    for i in range(0,n):
#        (ns[i:]) = ns_tmp(Index[i:])
#       
#    #Find the current best
#    nso = ns
#    Lighto = Lightn
#    nbest = ns[0,:]
#    Lightbest = Lightn[0]
#
#    # For output only
#    fbest = Lightbest
#
## move all fireflies to the better locations
#[ns] = ffa_move(n,d,ns,Lightn,nso,Lighto,nbest,Lightbest,alpha,betamin,gamma,Lb,Ub)
#
################################################
#
##All the subfunctions are listed here
##The initial locations of n fireflies
#
#def ([ns,Lightn] = init_ffa(n,d,Lb,Ub,u0)
#    for i in range Lb>0:
#        ns([i,:]) = Lb + (Ub-Lb) * rand(d)
#        else
#        #generate solutions around the random guess
#        for i in range(0,n):
#        (ns[i:]) = u0 + randn(d)
#        end
#        end
#        
##Initial value before function evaluations
#Lightn = ones(n)*10*100
#
##Move all fireflies toward brighter ones
#def ns = ffa_move(n, d, ns, Lightn, nso, Lighto, nbest, Lightbest, alpha, betamin, gamma,Lb, Ub))
#
#    #Scaling of the systems
#    scale = abs(Ub-Lb)
#    
#    #Updating fireflies
#    for i in range (0,n):
#        #The attractiveness parameter beta=exp(-gamma*r)
#        for j in range(0,n):
#            r = numpy.sqrt(numpy.sum((ns[i,:]-ns[j,:])**2));
#            #Update moves
#            if Lightn[i]>Lighto[j]: #Brighter and more attractive
#                beta0 = 1
#                beta = (beta0-betamin)*math.exp(-gamma*r**2)+betamin
#                tmpf = alpha*(numpy.random.rand(d)-0.5)*scale
#                ns[i,:] = ns[i,:]*(1-beta)+nso[j:]*beta+tmpf
#                
#    #Check if the update solutions/locations are within limits
#    [ns] = findlimits(n,ns,Lb,Ub)
#    
#    #This the  functon is optional
#
#def alpha_new(alpha,NGen):
#    delta = 1-(10**(-4)/0.9)**(1/NGen);
#    alpha = (1-delta)*alpha
#    return alpha
#    
#    #Make sure the fireflies are within the bounds/limits
#    
#    def [ns] = findlimits(n, ns, Lb, Ub)
#    for i in range (1:n)
#    #Apply the lower bound
#    ns_tmp = ns([i,:])
#    I = ns_tmp < Lb
#    ns_tmp(I) = Lb(I)
#    
#    
#    
#    #Appli the upper bounds
#    J = ns_tmp > Ub
#    ns_tmp(J) = Ub(J)
#    #Update this new move
#    ns(i,:) = ns_tmp
#    
#    
#    
#    ############### End of Firefly Algorithm implementation ################