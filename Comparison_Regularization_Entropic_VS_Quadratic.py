#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 17:36:22 2020

@author: theophanegregoir
"""
import cvxpy as cp
import os
import numpy as np
import matplotlib.pyplot as plt
from ProjectCOT import generate_D

#%% Folder management
experiment_folder="Entro_vs_Quadra_v8"
tol_diff = 0.01

if not os.path.isdir(experiment_folder):
    os.mkdir(experiment_folder)


def Sinkhorn(C,epsilon,f,niter = 500):    
    Err = np.zeros(niter)
    for it in range(niter):
        g = mina(C-f[:,None],epsilon)
        f = minb(C-g[None,:],epsilon)
        # generate the coupling
        P = a * np.exp((f[:,None]+g[None,:]-C)/epsilon) * b
        # check conservation of mass
        Err[it] = np.linalg.norm(np.sum(P,0)-b,1)
    return (P,Err)


def distmat(x,y):
    return np.sum(x**2,0)[:,None] + np.sum(y**2,0)[None,:] - 2*x.transpose().dot(y)

def mina_u(H,epsilon): return -epsilon*np.log( np.sum(a * np.exp(-H/epsilon),0) )
def minb_u(H,epsilon): return -epsilon*np.log( np.sum(b * np.exp(-H/epsilon),1) )

def mina(H,epsilon): return mina_u(H-np.min(H,0),epsilon) + np.min(H,0)
def minb(H,epsilon): return minb_u(H-np.min(H,1)[:,None],epsilon) + np.min(H,1)

experiment_completed = False
while not(experiment_completed) :
    
    #%% Number of points and weights
    tol_zero = 1e-6
    n = 100
    m = 150
    a = np.ones((n,1))/n
    b = np.ones((1,m))/m
    
    #%% Generate gaussian centered
    x = np.random.rand(2,n)-0.5
    
    #%% Generate circle data with
    theta = 2*np.pi*np.random.rand(1,m)
    r = 2.0 + .4*np.random.rand(1,m)
    y = np.vstack((np.cos(theta)*r,np.sin(theta)*r))
    
    #%% Distance and edges
    edges = []
    edge_cost = []
    
    for i in range(n):
        for j in range(m):
            edges.append((i, n+j))
            edge_cost.append((x[0,i]-y[0,j])**2+(x[1,i]-y[1,j])**2)
    
    
    #%% Data creation
    
    vertices = [i for i in range(n+m)]
    D = generate_D(vertices, edges) ### D in the article
    nb_edges = len(edges)
    nb_vertices = n+m
    
    initial_mass = np.zeros(nb_vertices) ### rho_0 in article
    initial_mass[:n] = 1./n
    final_mass = np.zeros(nb_vertices)
    final_mass[n:] = 1./m ### rho_1 in article
    f = final_mass - initial_mass ### f in article

    #%% Solving the unregularized problem
    
    J = cp.Variable(shape=(nb_edges,1), name="J")
    c = np.array(edge_cost).reshape((nb_edges,1))
    
    f = f.reshape((nb_vertices,1))
    
    constraints = [cp.matmul(D.T,J) == f, J >= 0]
    
    objective = cp.Minimize(cp.matmul(c.T, J))
    
    problem = cp.Problem(objective, constraints)
    
    solution_unreg = problem.solve()
    
    J_star_unreg = np.array(J.value)
    J_star_unreg = np.where(J_star_unreg < tol_zero, 0.0, J_star_unreg)
    
    P_unreg = np.zeros((n,m))
    
    for idx, edge in enumerate(edges):
        P_unreg[edge[0], edge[1]-n] = J_star_unreg[idx]


    #%% Solving the quadratic problem
    
    alpha = 0.5
    
    J = cp.Variable(shape=(nb_edges,1), name="J")
    c = np.array(edge_cost).reshape((nb_edges,1))
    
    f = f.reshape((nb_vertices,1))
    
    constraints = [cp.matmul(D.T,J) == f, J >= 0]
    
    objective = cp.Minimize(cp.matmul(c.T, J) + 0.1*alpha*cp.sum(J**2))
    
    problem = cp.Problem(objective, constraints)
    
    solution = problem.solve()
    
    
    J_star_quadra = np.array(J.value)
    
    J_star_quadra = np.where(J_star_quadra < tol_zero, 0.0, J_star_quadra)
    
    P_quadratic = np.zeros((n,m))
    
    for idx, edge in enumerate(edges):
        P_quadratic[edge[0], edge[1]-n] = J_star_quadra[idx]



    #%% Solving entropic
    N=[n,m]
    a = np.ones((n,1))/n
    b = np.ones((1,m))/m
    C = distmat(x,y)
    epsilon = 0.001
    
    (P_entropic,Err) = Sinkhorn(C,epsilon,np.zeros(n),20000)
    P_entropic = np.where(P_entropic < tol_zero, 0.0, P_entropic)
    
    
    eval_quadra = c.T @ J_star_quadra
    eval_entro = np.sum(P_entropic * C)
    
    dist_unreg_to_quadra = abs(eval_quadra[0][0] - solution_unreg)
    dist_unreg_to_entro = abs(eval_entro - solution_unreg)
    print("==============")
    print("DIFF")
    print(abs(dist_unreg_to_quadra - dist_unreg_to_entro) )
    if abs(dist_unreg_to_quadra - dist_unreg_to_entro) < tol_diff :
        experiment_completed = True
        
#%% Affichage RAW

plt.figure(figsize = (10,10), dpi=400)
plt.scatter(x[0, :], x[1, :], s=100, edgecolors="r", c='white')
plt.scatter(y[0, :], y[1, :], s=100, edgecolors="b", c='white')
plt.savefig(experiment_folder+"/"+"RAW_DATA.png", dpi=200)
plt.show()
plt.clf()

plt.figure(figsize = (10,10), dpi=400)
plt.scatter(x[0, :], x[1, :], s=100, edgecolors="r", c='white')
plt.scatter(y[0, :], y[1, :], s=100, edgecolors="b", c='white')
plt.axis("off")
plt.savefig(experiment_folder+"/"+"RAW_DATA_OFF.png", dpi=200)
plt.show()
plt.clf()

#%% Affichage Unreg

A = P_unreg
i,j = np.where(A != 0)

plt.figure(figsize = (10,10), dpi=400)
plt.scatter(x[0, :], x[1, :], s=100, edgecolors="r", c='white')
plt.scatter(y[0, :], y[1, :], s=100, edgecolors="b", c='white')
plt.plot([x[0,i],y[0,j]],[x[1,i],y[1,j]],'k',lw = 1)
plt.title("UNREG")
plt.savefig(experiment_folder+"/"+"UnregTransport.png", dpi=200)
plt.show()
plt.clf()

#OFF
plt.figure(figsize = (10,10), dpi=400)
plt.scatter(x[0, :], x[1, :], s=100, edgecolors="r", c='white')
plt.scatter(y[0, :], y[1, :], s=100, edgecolors="b", c='white')
plt.plot([x[0,i],y[0,j]],[x[1,i],y[1,j]],'k',lw = 1)
plt.title("UNREG")
plt.axis("off")
plt.savefig(experiment_folder+"/"+"UnregTransport_OFF.png", dpi=200)
plt.show()
plt.clf()

#%% Affichage Entropic

A = P_entropic
i,j = np.where(A != 0)

plt.figure(figsize = (10,10), dpi=400)
plt.scatter(x[0, :], x[1, :], s=100, edgecolors="r", c='white')
plt.scatter(y[0, :], y[1, :], s=100, edgecolors="b", c='white')
plt.plot([x[0,i],y[0,j]],[x[1,i],y[1,j]],'k',lw = 1)
plt.title("ENTRO")
plt.savefig(experiment_folder+"/"+"EntropicTransport.png", dpi=200)
plt.show()
plt.clf()

#OFF
plt.figure(figsize = (10,10), dpi=400)
plt.scatter(x[0, :], x[1, :], s=100, edgecolors="r", c='white')
plt.scatter(y[0, :], y[1, :], s=100, edgecolors="b", c='white')
plt.plot([x[0,i],y[0,j]],[x[1,i],y[1,j]],'k',lw = 1)
plt.title("ENTRO")
plt.axis("off")
plt.savefig(experiment_folder+"/"+"EntropicTransport_OFF.png", dpi=200)
plt.show()
plt.clf()

#%% Affichage Quadratic

A = P_quadratic
i,j = np.where(A != 0)

plt.figure(figsize = (10,10), dpi=400)
plt.scatter(x[0, :], x[1, :], s=100, edgecolors="r", c='white')
plt.scatter(y[0, :], y[1, :], s=100, edgecolors="b", c='white')
plt.plot([x[0,i],y[0,j]],[x[1,i],y[1,j]],'k',lw = 1)
plt.title("QUADRA")
plt.savefig(experiment_folder+"/"+"QuadraticTransport.png", dpi=200)
plt.show()
plt.clf()

#OFF
plt.figure(figsize = (10,10), dpi=400)
plt.scatter(x[0, :], x[1, :], s=100, edgecolors="r", c='white')
plt.scatter(y[0, :], y[1, :], s=100, edgecolors="b", c='white')
plt.plot([x[0,i],y[0,j]],[x[1,i],y[1,j]],'k',lw = 1)
plt.title("QUADRA")
plt.axis("off")
plt.savefig(experiment_folder+"/"+"QuadraticTransport_OFF.png", dpi=200)
plt.show()
plt.clf()

#%% plot of matrix

plt.figure(dpi=200)
plt.imshow(P_unreg)
plt.axis("off")
plt.savefig(experiment_folder+"/"+"P_unreg.png",dpi=200)
plt.show()
plt.clf()

plt.figure(dpi=200)
plt.imshow(P_entropic)
plt.axis("off")
plt.savefig(experiment_folder+"/"+"P_entropic.png",dpi=200)
plt.show()
plt.clf()

plt.figure(dpi=200)
plt.imshow(P_quadratic)
plt.axis("off")
plt.savefig(experiment_folder+"/"+"P_quadratic.png",dpi=200)
plt.show()
plt.clf()

#%% plot of histograms
nb_bins=50
right_l = 0.0075
top_l = 100000


plt.figure(dpi=400)
plt.hist(P_unreg.flatten(), bins=nb_bins, alpha=0.5, histtype='bar', ec='black')
plt.xlim(right = right_l)
plt.yscale('log', nonposy='clip')
plt.ylim(bottom = 1,top = top_l)
plt.savefig(experiment_folder+"/"+"histo_unreg.png",dpi=400)
plt.show()
plt.clf()

plt.figure(dpi=400)
plt.hist(P_entropic.flatten(),bins=nb_bins, alpha=0.5, histtype='bar', ec='black')
plt.xlim(right = right_l)
plt.yscale('log', nonposy='clip')
plt.ylim(bottom = 0,top = top_l)
plt.savefig(experiment_folder+"/"+"histo_entro.png",dpi=400)
plt.show()
plt.clf()

plt.figure(dpi=400)
plt.hist(P_quadratic.flatten(),bins=nb_bins, alpha=0.5, histtype='bar', ec='black')
plt.xlim(right = right_l)
plt.yscale('log', nonposy='clip')
plt.ylim(bottom = 1,top = top_l)
plt.savefig(experiment_folder+"/"+"histo_quadra.png",dpi=400)
plt.show()
plt.clf()


#%% Comparing distance
print("== DISTANCE ENTROPIC TO UNREG ==")
print(dist_unreg_to_entro)

print("== DISTANCE QUADRATIC TO UNREG ==")
print(dist_unreg_to_quadra)


