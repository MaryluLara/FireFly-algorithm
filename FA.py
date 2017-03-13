import math
import pylab
import numpy


# parameters [n N_iteration alpha betamin gamma]
para = [20, 500, 0.5, 0.2, 1]

# Simple bounds/limits for d-dimensional problems

d = 15

Lb = zeros(d)
Ub = ones(d)

#Initial random guess

u0 = Lb + (Ub-Lb) * rand(d)

[u, eval, numeval] = ffa_mincon(@cost, u0, Lb, Ub, para)

#Display results
bestsolution = u
bestojb = eval
total_numer_of_function_evaluations = numeval

#Cost or Objective function

function z = cost(x)
z= 20 + ((x**2)-10*cos(2*pi*x)) + ((y**2)-10*cos(2*pi*y))

######## Start FA ########

function [nbest, fbest, NumEval] = ffa_mincon(fhandle, u0, Lb, Ub, para)

#Check input  parameters (otherwise set as default values)

def myplot(x,y, para = [20, 500, 0.25, 0.20, 1], Ub=[], Lb=[]):
    print (Usuage FA_mincon(@cost, u0, Lb, Ub, para))
    
    
n = para(1)
MaxGeneration = para(2)
alpha = (3)
betamin = (4)
gamma = (5)


# Total number of function evaluations
NumEval = n*MaxGeneration
    


#Calculate dimension
d = size(u0)

#Initial  values of an array
zn = ones(n)*10**100


#generating the initial locations of n fireflies    
[ns,Lightn] = init_ffa(n,d,lb,Ub,u0)    
















# ranking fireflies by their light intensity/Objectives

[Lightn, Index] = msort(zn) 
ns_tmp =ns



#Find the current best
nso = ns
Lighto = Lightn

# For output only
fbest = Lightbest

# move all fireflies to the better locations
[ns] = ffa_move(n,d,ns,Lightn,nso,Lighto,nbest,Lightbest,alpha,betamin,gamma,Lb,Ub)

#Initial value before function evaluations
Lightn= ones(n)*10*100


#Scaling of the systems
scale = abs(Ub-Lb)