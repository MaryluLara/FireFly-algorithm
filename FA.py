import math
#import matplotlib.pyplot as plt
import numpy as np
import time

para = np.array([20, 500, 0.5, 0.2, 1])
convergence = []
n=para[0]  
MaxGeneration=para[1]
alpha=para[2]
betamin=para[3]
gamma=para[4]


# Numero total de Evaluaciones
NumEval=n*MaxGeneration

# Simple bounds/limits for d-dimensional problems
d = 20
#Lb = np.zeros(d)
#Ub = np.ones(d)

Lb = np.ones(d)*-5.12
Ub = np.ones(d)*5.12


zn=np.ones(n)
zn.fill(float("inf")) 

#ns(i,:)=Lb+(Ub-Lb).*rand(1,d);
ns=np.random.uniform(0,1,(n,d)) *(Ub-Lb)+Lb

Lightn=np.ones(n)
Lightn.fill(float("inf")) 
    
#[ns,Lightn]=init_ffa(n,d,Lb,Ub,u0)

def alpha_new(alpha, NGen):
    delta=1-(10**(-4)/0.9)**(1/NGen);
    alpha=(1-delta)*alpha
    return alpha

def objf(x):
    dim=len(x);
    z=np.sum(x**2-10*np.cos(2*math.pi*x))+10*dim
    return z


print("CS is optimizing  \""+objf.__name__+"\"")    
    
timerStart=time.time() 
startTime=time.strftime("%Y-%m-%d-%H-%M-%S")

# parameters [n N_iteration alpha betamin gamma]

for k in range(int(MaxGeneration)):
    #% This line of reducing alpha is optional
    alpha = alpha_new(alpha, MaxGeneration)# generando el vector de aleatoriedad alpha
    
    #% Evaluate new solutions (for all n fireflies)
    for i in range(int(n)):
        zn[i]=objf(ns[i,:])
        Lightn[i]=zn[i]
   
    # Ranking fireflies by their light intensity/objectives
    Lightn=np.sort(zn)
    Index=np.argsort(zn)
    ns=ns[Index,:]
    
    #Find the current best
    nso=ns
    Lighto=Lightn
    nbest=ns[0,:] 
    Lightbest=Lightn[0]
        
    #% For output only
    fbest=Lightbest;
  
    
   #% Move all fireflies to the better locations
#    [ns]=ffa_move(n,d,ns,Lightn,nso,Lighto,nbest,...
#          Lightbest,alpha,betamin,gamma,Lb,Ub);   
        
          
    scale=np.ones(d)*abs(Ub-Lb)
    for i in range (int(n)):
        # The attractiveness parameter beta=exp(-gamma*r)
        for j in range(int(n)):
            r=np.sqrt(np.sum((ns[i,:]-ns[j,:])**2));
            #r=1
            # Update moves
            if Lightn[i]>Lighto[j]: # Brighter and more attractive
               beta0=1
               beta=(beta0-betamin)*math.exp(-gamma*r**2)+betamin
               tmpf=alpha*(np.random.rand(d)-0.5)*scale
               ns[i,:]=ns[i,:]*(1-beta)+nso[j,:]*beta+tmpf        
              
                
                  
    #ns=numpy.clip(ns, lb, ub)
        
    convergence.append(fbest)
        	
    IterationNumber=k
    BestQuality=fbest
        
    if (k%1==0):
           print(['At iteration '+ str(k)+ ' the best fitness is '+ str(BestQuality)])
    #    
   ####################### End main loop 
   
timerEnd=time.time()  
endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
executionTime=timerEnd-timerStart
convergence=convergence

#print optimizer="FFA"
print objf.__name__      
print "Star", startTime
print "End", endTime
print "Total Time", executionTime             
                  
                       
                                
#    ############### End of Firefly Algorithm implementation ################