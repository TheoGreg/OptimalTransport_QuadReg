#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 23:30:22 2020

@author: theophanegregoir
"""
import os
import networkx as nx
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from ProjectCOT import create_new_edge, create_multiple_edges, generate_D, vizualize_graph, vizualize_bipartite_graph

#%% SECOND EXPERIMENT
experiment_folder="Quadra_behaviour"
if not os.path.isdir(experiment_folder):
    os.mkdir(experiment_folder)
    
#%% Variables of the problem as in article

nb_vertices = 10

vertices = [i for i in range(nb_vertices)] # V in the article (each vertex corresponds to a number)

edges = []

for i in range(nb_vertices//2):
    for j in range(nb_vertices//2, nb_vertices):
        edges.append((i,j))

nb_edges = len(edges)

edge_cost = list(np.random.randint(low=1, high=10, size=nb_edges))

D = generate_D(vertices, edges) ### D in the article
initial_mass = np.zeros(nb_vertices) ### rho_0 in article
initial_mass[:(nb_vertices//2)] = np.random.random(size=nb_vertices//2)
initial_mass[:(nb_vertices//2)] = initial_mass[:(nb_vertices//2)] / np.sum(initial_mass[:(nb_vertices//2)])
final_mass = np.zeros(nb_vertices)
final_mass[(nb_vertices//2):] = np.random.random(size=nb_vertices//2)
final_mass[(nb_vertices//2):] = final_mass[(nb_vertices//2):] / np.sum(final_mass[(nb_vertices//2):])
f = final_mass - initial_mass ### f in article


#%% solving for different alphas

alphas = []
min_pow = 0
max_pow = 3
alphas+=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for power in range(min_pow, max_pow):
    alphas += [i * 10**(power) for i in range(1, 10)]

alphas.append(10**(max_pow))

cost_part = []
l2_part = []

for alpha in alphas :
    J = cp.Variable(shape=(nb_edges,1), name="J")
    c = np.array(edge_cost).reshape((nb_edges,1))
    f = f.reshape((nb_vertices,1))
    constraints = [cp.matmul(D.T,J) == f, J >= 0]
    objective = cp.Minimize(cp.matmul(c.T, J) + 0.5*alpha*cp.sum(J**2))
    problem = cp.Problem(objective, constraints)
    solution = problem.solve()
    J_quadra = np.array(J.value)
    eval_quadra = c.T @ J_quadra
    cost_part.append(eval_quadra[0][0])
    l2_part.append(np.sum(J_quadra**2))
    
#%% Plot

plt.figure(figsize=(20,8), dpi=200)
fig, ax1 = plt.subplots()
ax1.set_xlabel(r'$\alpha$', fontsize = 18)
plt.xticks(fontsize=14)
ax1.set_xscale('log')
ax1.set_xlim(left = max(alphas), right = min(alphas))
ax1.set_ylabel(r'$c^{t}J_{\alpha}$', color='b', fontsize = 20, rotation=90)
ax1.set_ylim(top = 6.5, bottom = 4.5)
plt.yticks([4.5,5.0,5.5,6.0,6.5],fontsize=14)
ax1.plot(alphas, cost_part, color='b',alpha=0.8)

ax2 = ax1.twinx()  
ax2.set_ylabel(r'$J_{\alpha}^{t} \: J_{\alpha}$', color='r', fontsize = 20, rotation=90)  # we already handled the x-label with ax1
ax2.set_ylim(top = 0.12, bottom = 0.05)
plt.yticks([0.05,0.075,0.10,0.125],fontsize=14)
ax2.plot(alphas, l2_part, color='r',alpha=0.8)
plt.savefig(experiment_folder+'/'+"behaviour_plot.png", dpi=400, bbox_inches='tight')

plt.show()

# plt.plot(alphas, l2_part, color='r',alpha=0.8, label="$J^{t}J$")
# plt.xscale('log')
# #plt.yscale('log')
# #plt.yticks([1.0,1e-1,1e-2], fontsize=16)
# #plt.ylabel(r'$\|\|  \: J^{*} - J_{quadra}  \: \|\|$', fontsize=18)
# plt.xlim(left = max(alphas), right = 1e-1)
# plt.xlabel(r'$\alpha$', fontsize=20)
# plt.xticks(fontsize=16)
# plt.savefig(experiment_folder+'/'+"distances_variables_plot.png", dpi=200)
# plt.show()
plt.clf()


