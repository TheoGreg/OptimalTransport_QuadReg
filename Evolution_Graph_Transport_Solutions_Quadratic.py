#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 10:50:04 2020

@author: theophanegregoir
"""
import os
import networkx as nx
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from ProjectCOT import create_new_edge, create_multiple_edges, generate_D, vizualize_graph, vizualize_bipartite_graph

#%% Folder management
experiment_folder="Quadra_to_Unreg_v3"

if not os.path.isdir(experiment_folder):
    os.mkdir(experiment_folder)

#%% Vizualize
#vizualize_bipartite_graph(vertices, edges, edge_cost)

#%% Defining primal LP problem

alphas = [0.001,0.01,0.1,0.5,1.,10.]

dico_alphas = {}
for alpha in alphas :
    dico_alphas[alpha] = {}

### Solve various quadratic depending on alpha

vertices = [i for i in range(8)]
nb_vertices = len(vertices)
pos_points = []

### Starting node : 0
pos_points.append([1.,2.])

### first row : 1,2,3
pos_points.append([3.,1.5])
pos_points.append([3.,2.])
pos_points.append([3.,2.5])

### second row : 4,5,6
pos_points.append([5.,1.5])
pos_points.append([5.,2.])
pos_points.append([5.,2.5])

### end node : 7
pos_points.append([7.,2.])

x = np.array(pos_points).T
print(x.shape)


#%% Distance and edges
tol_zero=1e-5
edges = []
edge_cost = []

for i in range(len(vertices)):
    if i == 0 :
        for j in range(1, 4):
            edges.append((i, j))
            edge_cost.append((pos_points[i][0]-pos_points[j][0])**2+(pos_points[i][1]-pos_points[j][1])**2)
    elif i <= 3 :
        for j in range(4,7):
            edges.append((i, j))
            edge_cost.append((pos_points[i][0]-pos_points[j][0])**2+(pos_points[i][1]-pos_points[j][1])**2)
    elif i <= 6 :
        edges.append((i, 7))
        edge_cost.append((pos_points[i][0]-pos_points[7][0])**2+(pos_points[i][1]-pos_points[7][1])**2)

nb_edges = len(edges)
D = generate_D(vertices, edges) ### D in the article
initial_mass = np.zeros(nb_vertices) ### rho_0 in article
initial_mass[0] = 1.
final_mass = np.zeros(nb_vertices)
final_mass[-1] = 1. ### rho_1 in article
f = final_mass - initial_mass ### f in article

J = cp.Variable(shape=(nb_edges,1), name="J")
c = np.array(edge_cost).reshape((nb_edges,1))
f = f.reshape((nb_vertices,1))
constraints = [cp.matmul(D.T,J) == f, J >= 0]
objective = cp.Minimize(cp.matmul(c.T, J))
problem = cp.Problem(objective, constraints)
solution_unreg = problem.solve()
J_unreg = np.array(J.value)

for alpha in alphas :
    J = cp.Variable(shape=(nb_edges,1), name="J")
    c = np.array(edge_cost).reshape((nb_edges,1))
    f = f.reshape((nb_vertices,1))
    constraints = [cp.matmul(D.T,J) == f, J >= 0]
    objective = cp.Minimize(cp.matmul(c.T, J) + 0.5*alpha*cp.sum(J**2))
    problem = cp.Problem(objective, constraints)
    solution = problem.solve()
    J_quadra = np.array(J.value)
    J_quadra = np.where(J_quadra < tol_zero, 0.0, J_quadra)
    eval_quadra = c.T @ J_quadra
    
    dico_alphas[alpha]['eval'] =abs(eval_quadra[0][0] - solution_unreg)
    
    dico_alphas[alpha]['J']= J_quadra
    
    P_quadratic = np.zeros((nb_vertices, nb_vertices))
    
    for idx, edge in enumerate(edges):
        P_quadratic[edge[0], edge[1]] = J_quadra[idx]
    
    dico_alphas[alpha]['P'] = P_quadratic
    
# for alpha in alphas :
#     distances_variables_mean.append(np.mean(dico_alphas[alpha]['var']))
#     distances_variables_std.append(np.std(dico_alphas[alpha]['var']))
#     distances_eval_mean.append(np.mean(dico_alphas[alpha]['eval']))
#     distances_eval_std.append(np.std(dico_alphas[alpha]['eval']))
        
    
#%% PLOTTING


plt.figure(figsize=(20,20), dpi=400)
alphas.reverse()
for k, alpha in enumerate(alphas):
    A = dico_alphas[alpha]['P']
    i,j = np.where(A != 0)
    plt.subplot(int(len(alphas)/2),2,k+1)
    plt.scatter(x[0,0], x[1,0], s=100, edgecolors="r", c='r')
    plt.scatter(x[0,-1], x[1,-1], s=100, edgecolors="g", c='g')
    plt.scatter(x[0, 1:-1], x[1, 1:-1], s=100, edgecolors="b", c='white')
    plt.plot([x[0,i],x[0,j]],[x[1,i],x[1,j]],'k',lw = 1)
    plt.yticks([1.0, 1.5, 2.0, 2.5, 3.0], fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.title(r'$\alpha$' + " = " + str(alpha), fontsize= 20)
plt.savefig("GRAPH_output_v2.png", dpi=400)
plt.show()
plt.clf()

plt.figure(figsize=(10,8), dpi=400)
A = dico_alphas[10]['P']
i,j = np.where(A != 0)
plt.scatter(x[0,0], x[1,0], s=100, edgecolors="r", c='r')
plt.scatter(x[0,-1], x[1,-1], s=100, edgecolors="g", c='g')
plt.scatter(x[0, 1:-1], x[1, 1:-1], s=100, edgecolors="b", c='white')
plt.plot([x[0,i],x[0,j]],[x[1,i],x[1,j]],'k',lw = 1)
plt.yticks([1.0, 1.5, 2.0, 2.5, 3.0], fontsize = 18)
plt.xticks(fontsize = 18)
plt.savefig("INITIAL_GRAPH.png", dpi=400)