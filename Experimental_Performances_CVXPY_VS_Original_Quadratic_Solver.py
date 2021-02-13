#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 21:15:59 2020

@author: theophanegregoir
"""
import time
import cvxpy as cp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from choldate import cholupdate, choldowndate


#%% Graph utils

def create_new_edge(vertices, edges, edge_cost, start_vertex=None, end_vertex=None, link_cost=None):
    ### Chosen edge to add
    if (start_vertex!=None and end_vertex!=None) and link_cost!=None :
        #print("Adding chosen edge")
        edges.append((start_vertex,end_vertex))
        edge_cost.append(link_cost)
        
    ### Random edge to add
    else :
        #print("Adding random edge")
        max_iter = 2000000
        search_edge = True
        it = 0
        while search_edge and it < max_iter :
            
            ### select 2 different nodes
            node1 = np.random.choice(vertices)  
            node2 = np.random.choice(vertices)       
            while node2 == node1 :
                node2 = np.random.choice(vertices) 
            
            ### check if edge already exists
            if not((node1, node2) in edges):
                edges.append((node1,node2))
                edge_cost.append(np.random.random())
                search_edge = False
            
            it+=1
            
    return edges, edge_cost

def create_multiple_edges(nb_to_add, vertices, edges, edge_cost):
    for k in range(nb_to_add):
        edges, edge_cost = create_new_edge(vertices, edges, edge_cost)
    return edges, edge_cost

def create_all_edges(vertices):
    edges=[]
    for k in range(len(vertices)):
        for p in range(len(vertices)):
            if p!=k :
                edges.append((k,p))
    return edges

def create_bipartite_edges(vertices):
    edges_local = []
    edges_cost_local = []
    separation_vertex = int(len(vertices)/2)
    for k in range(separation_vertex):
        for p in range(separation_vertex,len(vertices)):
            edges_local.append((k,p))
            edges_cost_local.append(10*np.random.random())
    return edges_local, edges_cost_local

def generate_D(vertices, edges):
    D = np.zeros((len(edges), len(vertices)))
    for idx, edge in enumerate(edges):
        D[idx, edge[0]] = -1.0
        D[idx, edge[1]] = 1.0
    return D

def generate_mass(vertices):
    p = np.random.random(len(vertices))
    p = p / np.sum(p)
    return p

def generate_problem(vertices, portion_source):

    nb_vertices = len(vertices)
    initial_mass = np.zeros(nb_vertices) ### rho_0 in article
    final_mass = np.zeros(nb_vertices)

    nb_sources = int(nb_vertices * portion_source)
    p_source = np.random.random(nb_sources)
    p_source = p_source / np.sum(p_source)
    initial_mass[:nb_sources] = p_source
    
    p_sink = np.random.random(nb_sources)
    p_sink = p_sink / np.sum(p_sink)
    final_mass[-nb_sources:] = p_sink
    
    return initial_mass, final_mass

def generate_bipartite_problem(vertices):
    
    nb_vertices = len(vertices)
    initial_mass = np.zeros(nb_vertices) ### rho_0 in article
    final_mass = np.zeros(nb_vertices)
    
    separation_vertex = int(len(vertices)/2)
    
    p_source = np.random.random(separation_vertex)
    p_source = p_source / np.sum(p_source)
    initial_mass[:separation_vertex] = p_source
    
    p_sink = np.random.random(len(vertices) - separation_vertex)
    p_sink = p_sink / np.sum(p_sink)
    final_mass[separation_vertex:] = p_sink
    return initial_mass, final_mass
    

def vizualize_graph(vertices, edges, edge_cost):
    DG = nx.DiGraph()
    DG.add_nodes_from(vertices)
    for k in range(len(edges)):
        DG.add_weighted_edges_from([(edges[k][0],edges[k][1],round(edge_cost[k],3))])
    
    pos={i:(np.random.randint(0,50),np.random.randint(0,100)) for i in range(len(vertices))}
    nx.draw(DG, pos, with_labels=True)
    labels = nx.get_edge_attributes(DG,'weight')
    nx.draw_networkx_edge_labels(DG,pos,edge_labels=labels, label_pos=0.75)

def vizualize_bipartite_graph(vertices, edges, edge_cost):
    DG = nx.DiGraph()
    DG.add_nodes_from(vertices)
    for k in range(len(edges)):
        DG.add_weighted_edges_from([(edges[k][0],edges[k][1],round(edge_cost[k],3))])
    
    first_part = nx.bipartite.sets(DG)[0]
    
    pos=nx.drawing.layout.bipartite_layout(DG, first_part)
    nx.draw(DG, pos, with_labels=True)
    labels = nx.get_edge_attributes(DG,'weight')
    nx.draw_networkx_edge_labels(DG,pos,edge_labels=labels,label_pos=0.75,font_weight=2., font_size=14)

def graph_components(vertices, edges):
    adj = [[] for i in range(len(vertices))]
    for edg in edges :
        adj[edg[0]].append(edg[1])
        adj[edg[1]].append(edg[0])
        
    def DFSUtil(vertices, adj, temp, v, visited):
 
        # Mark the current vertex as visited
        visited[v] = True
 
        # Store the vertex to list
        temp.append(v)
 
        # Repeat for all vertices adjacent
        # to this vertex v
        for i in adj[v]:
            if visited[i] == False:
                 # Update the list
                temp = DFSUtil(vertices, adj, temp, i, visited)
        
        return temp
    
    visited = []
    cc = []
    for i in range(len(vertices)):
        visited.append(False)
    for v in range(len(vertices)):
        if visited[v] == False:
            temp = []
            cc.append(DFSUtil(vertices, adj, temp, v, visited))
    
    return cc
    
def generate_N(vertices, act_edges):
    N = np.array([], dtype=float).reshape(len(vertices),0)
    comps = graph_components(vertices, act_edges)
    for compo in comps :
        col = np.array([np.sqrt(1./len(compo)) if (i in compo) else 0. for i in range(len(vertices))]).reshape(len(vertices),1)
        N = np.concatenate([N, col], axis = 1) if N.shape[1]!=0 else col #np.concatenate((N, col), axis=1)
    return N
    
#%% Algorithm from article

def RegularizedDualTransport(vertices, edges, edge_cost, alpha, f, tol):
    
    ### time management
    start = time.time()
    
    ### trackers
    t_list = []
    grad_norm = []
    
    # print("Initial mass : " + str(initial_mass)) 
    # print("Target mass : " + str(final_mass)) 
    
    ### main dimensions of the problem
    nb_v = len(vertices)
    nb_e = len(edges)
    
    ### edge cost to np.array
    edge_cost = np.array(edge_cost)
    
    ### Initialize randomly dual Variable 
    p = 10*np.random.random(size=(nb_v,))
    
    ### Generate D 
    D = generate_D(vertices, edges)
    
    ### Compute v and M
    v = np.dot(D,p) - np.array(edge_cost)
    # print("v")
    # print(v)
    active_edges = np.array(v > 0.0,dtype=float)
    M = np.diag(active_edges.reshape((nb_e,)))
    
    ### Compute L
    L = D.T @ M @ D
    
    ### Compute N : basis of null space of L
    act_edges = []
    for idx in range(len(active_edges)) :
        if int(active_edges[idx]) == 1 :
            act_edges.append(edges[idx])
    N = generate_N(vertices, act_edges)
    
    # print("INITIAL ACTIVE EDGES : " + str(act_edges))
    
    ### QR decomposition
    upper_part = M @ D
    lower_part = N.T
    W = np.concatenate((upper_part, lower_part), axis=0)
    Q, R = np.linalg.qr(W)
    
    max_iter = 3000
    converged = False
    
    for k in range(max_iter):
        
        nb_iter = int(k)
        # print("===============")
        # print("STEP " + str(k))
        
        ### Choose search direction
        s = alpha * f - D.T @ M @ v
        # print("s : " + str(s))
        grad_norm.append(np.linalg.norm(s))
        
        if np.linalg.norm(s) < tol :
            optimal_p = p
            #print("OPTIMAL VALUE REACHED BEFORE MAX ITERATION !")
            converged = True
            break
        
        # if k%2 == 1 :
        #     print("INV R : " + str(np.linalg.inv(R)))    
        #     print("N : " + str(N))
        #     print("NNT : " + str(N @ N.T))
        #     s = np.linalg.inv(R) @ np.linalg.inv(R).T @ (s - (N @ N.T @ s))
        
        # print("s : " + str(s))
    
        ### Line search using direction      
        t_quadra = (((alpha * f).T @ s) - (v.T@M@D@s))/(s.T@(L@s))
        # print("t_quadra : " + str(t_quadra))
        
        ### Search for t_active_sets : how to keep M constant
        h = - v / (D@s)
        
        t_active_set = np.where(h > 0, h, np.inf).min()
        # print("t_active_set : " + str(t_active_set))
        
        ### final parameter kept
        t = min(t_quadra,t_active_set)
        t_list.append(t)
        # print("t : " + str(t))
        
        ### update dual variable
        p = p + t * s
        # print("p : " + str(p))
        
        ### Store previous step
        previous_v = v.copy()
        previous_active_edges = active_edges.copy()
        previous_M = M.copy()
        
        ### Compute new M and v
        v = np.dot(D,p) - np.array(edge_cost)
        active_edges = np.array(v > 0.0,dtype=float)
        M = np.diag(active_edges.reshape((nb_e,)))
        
        ### Difference between active edges
        diff_active = active_edges - previous_active_edges
        idx_edges_removed = list(np.where(diff_active == -1.)[0])
        idx_edges_added = list(np.where(diff_active == 1.)[0])
        
        
        updated_edges = []
        for idx in range(len(previous_active_edges)) :
            if int(previous_active_edges[idx]) == 1 :
                updated_edges.append(edges[idx])
        
        ### New active edges to add
        for added in list(idx_edges_added) :
            # print("ADDING EDGE " + str(edges[added]))
            updated_edges.append(edges[added])
            
            L = L + np.outer(D[added],D[added].T)
            # cholupdate(R,D[added])
            
            ##Check if two components are merging
            start_vertex = edges[added][0]
            end_vertex = edges[added][1]
            
            component_start_vertex = int(np.where(np.abs(N[start_vertex,:]) > 1e-12)[0][0])
            component_end_vertex = int(np.where(np.abs(N[end_vertex,:]) > 1e-12)[0][0])
            
            ## Two components are merged
            if component_start_vertex != component_end_vertex:
                
                # print("TWO COMPONENTS ARE MERGING")
                
                members_of_start_compo = list(np.where(np.abs(N[:,component_start_vertex]) > 1e-12)[0])
                members_of_end_compo = list(np.where(np.abs(N[:,component_end_vertex]) > 1e-12)[0])
                ### remove two columns of the components
                start_compo = N[:,component_start_vertex].copy()
                end_compo = N[:,component_end_vertex].copy()
                
                # print("N before : ")
                # print(N)
                N = np.delete(N, [component_start_vertex, component_end_vertex], 1)
                
                ### Defining new merged component
                members_new_compo = members_of_start_compo + members_of_end_compo
                new_compo = np.array([np.sqrt(1./len(members_new_compo)) if (i in members_new_compo) else 0. for i in range(nb_v)]).reshape(nb_v,1)
                
                ### Adding new merged column
                N = np.concatenate((N, new_compo), axis=1)
                # print("N after : ")
                # print(N)
                
                ### Corresponding Cholupdates
                # cholupdate(R,new_compo.transpose().reshape(nb_v,)) 
                # choldowndate(R,start_compo)
                # choldowndate(R,end_compo)
            
        
        ### Old active edges to remove
        for removed in list(idx_edges_removed) :            
            # print("REMOVING EDGE " + str(edges[removed]))
            updated_edges.remove(edges[removed])
            L = L - np.outer(D[removed],D[removed].T)
            
            ### Check if two components are splitting
            # Find joint component
            start_vertex = edges[removed][0]
            end_vertex = edges[removed][1]
            joint_component_idx = int(np.where(np.abs(N[start_vertex,:]) > 1e-12)[0][0])
            joint_compo = N[:,joint_component_idx].copy()
            
            ### Using DFS to see if the components are splitting
            # print("UPDATED EDGES")
            # print(updated_edges)
            all_components = graph_components(vertices, updated_edges)
            # print("Components of the graph after removal")
            # print(all_components)
            splitted_components = True
            for compo in all_components :
                if (start_vertex in compo) and (end_vertex in compo) :
                    splitted_components = False
                    break
                elif (start_vertex in compo) :
                    new_compo_start = np.array([np.sqrt(1./len(compo)) if (i in compo) else 0. for i in range(nb_v)]).reshape(nb_v,1)
                elif (end_vertex in compo) :
                    new_compo_end = np.array([np.sqrt(1./len(compo)) if (i in compo) else 0. for i in range(nb_v)]).reshape(nb_v,1)
                    
            if splitted_components :
                # print("COMPONENT IS SPLITTING")
                # print("N before : ")
                # print(N)   
                ### Updates on N
                ### Deleting old component
                N = np.delete(N, joint_component_idx, 1)
                ### Adding first new column
                N = np.concatenate((N, new_compo_start), axis=1)
                ### Adding second new column
                N = np.concatenate((N, new_compo_end), axis=1)
                # print("N after : ")
                # print(N)
                
                ### Cholupdate on R
                # cholupdate(R,new_compo_start.transpose().reshape(nb_v,))
                # cholupdate(R,new_compo_end.transpose().reshape(nb_v,))
                # choldowndate(R,joint_compo)
            
            ### Downdate on R
            # choldowndate(R,D[removed])
        
        R = np.linalg.cholesky(L + N @ N.T).T
        # print("UPDATED ACTIVE EDGES : " + str(updated_edges))
    
    if not(converged):
        #print("MAX ITERATIONS REACHED")
        optimal_p = p
    
    #%% Back to primal problem             
    
    # print("Back to primal : ")
    J = np.maximum(list(D @ optimal_p - np.array(edge_cost)), list(np.zeros(nb_e))) / alpha
    # print(J)
        
    runtime = time.time() - start
    
    return t_list, grad_norm, runtime, nb_iter, J 


def RegularizedDualTransport_classic(vertices, edges, edge_cost, alpha, f):
    start = time.time()
    nb_edges = len(edges)
    nb_vertices = len(vertices)

    J = cp.Variable(shape=(nb_edges,1), name="J")
    c = np.array(edge_cost).reshape((nb_edges,1))
    D = generate_D(vertices, edges)
    f = f.reshape((nb_vertices,1))
    
    constraints = [cp.matmul(D.T,J) == f, J >= 0]
    
    objective = cp.Minimize(cp.matmul(c.T, J) + 0.5*alpha*cp.sum(J**2))
    
    problem = cp.Problem(objective, constraints)
    
    solution = problem.solve(verbose=False)
    
    nb_iter = problem.solver_stats.num_iters
    
    # print(solution)
    
    J_star = np.array(J.value).reshape((nb_edges,))
    
    # print(J_star)
    
    runtime = time.time() - start
    
    return runtime, nb_iter, J_star

#%% Experiment parameters :
    
alphas = []
min_pow = -1
max_pow = 2
alphas.append(5 * 10**(-2))
for power in range(min_pow, max_pow):
    alphas += [1 * 10**(power), 5 * 10**(power)]

alphas.append(10**(max_pow))

vertices_parameters = [4, 6, 8, 10]

def experiment_fullyconnected_graph(alphas, vertices_parameters):
    
    nb_repeat = 100
    tol = 1e-5
    
    dico_results = {}
    for method in ["CVXPY", "OURS"]:
        dico_results[method]={}
        for nb_v in vertices_parameters:
            dico_results[method][nb_v] = {}
            for alpha in alphas :
               dico_results[method][nb_v][alpha] = {} 
               dico_results[method][nb_v][alpha]['ERROR'] = 0
               dico_results[method][nb_v][alpha]['Runtime'] = []
               dico_results[method][nb_v][alpha]['nb_iter'] = []
               dico_results[method][nb_v][alpha]['distance'] = []
           
    for step in range(nb_repeat):
        print("=========")
        print("STEP " + str(step+1) + " - FULL")
        for nb_v in vertices_parameters:
            nb_e = nb_v*(nb_v - 1)
            vertices = [i for i in range(nb_v)] # V in the article (each vertex corresponds to a number)
            edges = create_all_edges(vertices)
            edge_cost = list(10*np.random.random(size=nb_e))
            initial_mass = np.zeros(nb_v) ### rho_0 in article
            final_mass = np.zeros(nb_v)
            initial_mass, final_mass = generate_problem(vertices, portion_source = 0.4)
            f = final_mass - initial_mass ### f in article
            for alpha in alphas :
                
                runtime_classic, nb_iter_classic, J_classic = RegularizedDualTransport_classic(vertices, edges, edge_cost, alpha, f)
                
                t_list, grad_norm, runtime, nb_iter, J_ours  = RegularizedDualTransport(vertices, edges, edge_cost, alpha, f, tol)
                
                distance_sol = np.linalg.norm(J_classic.reshape(nb_e,) - J_ours.reshape(nb_e,))
                
                if distance_sol > 0.1 :
                    #print("ERROR")
                    dico_results["OURS"][nb_v][alpha]['ERROR'] += 1
                else :
                    #print("VALID")
                    dico_results["CVXPY"][nb_v][alpha]['Runtime'].append(runtime_classic)
                    dico_results["CVXPY"][nb_v][alpha]['nb_iter'].append(nb_iter_classic)
                    dico_results["OURS"][nb_v][alpha]['Runtime'].append(runtime)
                    dico_results["OURS"][nb_v][alpha]['nb_iter'].append(nb_iter)
                    dico_results["OURS"][nb_v][alpha]['distance'].append(distance_sol)
    
    return dico_results


def experiment_bipartite_graph(alphas,vertices_parameters):
    
    nb_repeat = 100
    tol = 1e-5
    
    dico_results = {}
    for method in ["CVXPY", "OURS"]:
        dico_results[method]={}
        for nb_v in vertices_parameters:
            dico_results[method][nb_v] = {}
            for alpha in alphas :
               dico_results[method][nb_v][alpha] = {} 
               dico_results[method][nb_v][alpha]['ERROR'] = 0
               dico_results[method][nb_v][alpha]['Runtime'] = []
               dico_results[method][nb_v][alpha]['nb_iter'] = []
               dico_results[method][nb_v][alpha]['distance'] = []
           
    for step in range(nb_repeat):
        print("=========")
        print("STEP " + str(step+1) + " - BIP")
        for nb_v in vertices_parameters:
            vertices = [i for i in range(nb_v)] # V in the article (each vertex corresponds to a number)
            edges, e_c = create_bipartite_edges(vertices)
            nb_e = len(edges)
            edge_cost = list(10*np.random.random(size=nb_e))
            initial_mass, final_mass = generate_bipartite_problem(vertices)
            f = final_mass - initial_mass ### f in article
            for alpha in alphas :
                print(alpha)
                runtime_classic, nb_iter_classic, J_classic = RegularizedDualTransport_classic(vertices, edges, edge_cost, alpha, f)
                
                t_list, grad_norm, runtime, nb_iter, J_ours  = RegularizedDualTransport(vertices, edges, edge_cost, alpha, f, tol)
                
                distance_sol = np.linalg.norm(J_classic.reshape(nb_e,) - J_ours.reshape(nb_e,))
                
                if distance_sol > 0.1 :
                    #print("ERROR")
                    dico_results["OURS"][nb_v][alpha]['ERROR'] += 1
                else :
                    #print("VALID")
                    dico_results["CVXPY"][nb_v][alpha]['Runtime'].append(runtime_classic)
                    dico_results["CVXPY"][nb_v][alpha]['nb_iter'].append(nb_iter_classic)
                    dico_results["OURS"][nb_v][alpha]['Runtime'].append(runtime)
                    dico_results["OURS"][nb_v][alpha]['nb_iter'].append(nb_iter)
                    dico_results["OURS"][nb_v][alpha]['distance'].append(distance_sol)
    
    return dico_results
                    

#%% Save fully connected results :
#dico_fully_connected = experiment_fullyconnected_graph(alphas,vertices_parameters)
import pickle
#pickle.dump(dico_fully_connected, open('fully_connected_perf_v2.p', "wb" ))
dico_fully_connected = pickle.load(open('fully_connected_perf_v2.p', "rb" ))

#%% Save bipartite results

#dico_bipartite = experiment_bipartite_graph(alphas,vertices_parameters)
import pickle
#pickle.dump(dico_bipartite, open('bipartite_perf_v2.p', "wb" ))
dico_bipartite = pickle.load(open('bipartite_perf_v2.p', "rb" ))

#%% Analyse results
import pickle
dico_fully_connected = pickle.load(open('fully_connected_perf_v2.p', "rb" ))
dico_bipartite = pickle.load(open('bipartite_perf_v2.p', "rb" ))

### FULLY CONNECTED
import pandas as pd
dico_failure_full = {}
vertices_parameters = list(dico_fully_connected["CVXPY"].keys())

for nb_v in vertices_parameters:
    dico_failure_full[nb_v] = {}
    for alpha in alphas :
        dico_failure_full[nb_v][alpha] = 100 * ( float(dico_fully_connected["OURS"][nb_v][alpha]['ERROR']/100.))

df_failure_full = pd.DataFrame(data=dico_failure_full)

dico_iter_full_ours = {}

for nb_v in vertices_parameters:
    dico_iter_full_ours[nb_v] = {}
    for alpha in alphas :
        dico_iter_full_ours[nb_v][alpha] = np.median(dico_fully_connected["OURS"][nb_v][alpha]['nb_iter'])

df_iter_full_ours = pd.DataFrame(data=dico_iter_full_ours)

dico_iter_full_CVX = {}

for nb_v in vertices_parameters:
    dico_iter_full_CVX[nb_v] = {}
    for alpha in alphas :
        dico_iter_full_CVX[nb_v][alpha] = np.median(dico_fully_connected["CVXPY"][nb_v][alpha]['nb_iter'])

df_iter_full_CVXPY = pd.DataFrame(data=dico_iter_full_CVX)

### RUNTIME

dico_runtime_full_ours = {}

for nb_v in vertices_parameters:
    dico_runtime_full_ours[nb_v] = {}
    for alpha in alphas :
        dico_runtime_full_ours[nb_v][alpha] = np.median(dico_fully_connected["OURS"][nb_v][alpha]['Runtime'])

df_runtime_full_ours = pd.DataFrame(data=dico_runtime_full_ours)

dico_runtime_full_CVX = {}

for nb_v in vertices_parameters:
    dico_runtime_full_CVX[nb_v] = {}
    for alpha in alphas :
        dico_runtime_full_CVX[nb_v][alpha] = np.median(dico_fully_connected["CVXPY"][nb_v][alpha]['Runtime'])

df_runtime_full_CVXPY = pd.DataFrame(data=dico_runtime_full_CVX)

### BIPARTITE
dico_failure_bip = {}
vertices_parameters = list(dico_bipartite["CVXPY"].keys())

for nb_v in vertices_parameters:
    dico_failure_bip[nb_v] = {}
    for alpha in alphas :
        dico_failure_bip[nb_v][alpha] = 100 * (float(dico_bipartite["OURS"][nb_v][alpha]['ERROR']/100.))

df_failure_bip = pd.DataFrame(data=dico_failure_bip)

dico_iter_bipartite_ours = {}

for nb_v in vertices_parameters:
    dico_iter_bipartite_ours[nb_v] = {}
    for alpha in alphas :
        dico_iter_bipartite_ours[nb_v][alpha] = np.median(dico_bipartite["OURS"][nb_v][alpha]['nb_iter'])

df_iter_bipartite_ours = pd.DataFrame(data=dico_iter_bipartite_ours)

dico_iter_bipartite_CVX = {}

for nb_v in vertices_parameters:
    dico_iter_bipartite_CVX[nb_v] = {}
    for alpha in alphas :
        dico_iter_bipartite_CVX[nb_v][alpha] = np.median(dico_bipartite["CVXPY"][nb_v][alpha]['nb_iter'])

df_iter_bipartite_CVXPY = pd.DataFrame(data=dico_iter_bipartite_CVX)

### RUNTIME

dico_runtime_bipartite_ours = {}

for nb_v in vertices_parameters:
    dico_runtime_bipartite_ours[nb_v] = {}
    for alpha in alphas :
        dico_runtime_bipartite_ours[nb_v][alpha] = np.median(dico_bipartite["OURS"][nb_v][alpha]['Runtime'])

df_runtime_bipartite_ours = pd.DataFrame(data=dico_runtime_bipartite_ours)

dico_runtime_bipartite_CVX = {}

for nb_v in vertices_parameters:
    dico_runtime_bipartite_CVX[nb_v] = {}
    for alpha in alphas :
        dico_runtime_bipartite_CVX[nb_v][alpha] = np.median(dico_bipartite["CVXPY"][nb_v][alpha]['Runtime'])

df_runtime_bipartite_CVXPY = pd.DataFrame(data=dico_runtime_bipartite_CVX)


# plt.figure(figsize=(10,20), dpi=200)
# plt.subplot(2,1,1)
# vizualize_bipartite_graph(vertices, edges, edge_cost)
# plt.subplot(2,1,2)
# vizualize_bipartite_graph(vertices, edges, list(edges_optimal_transfer.reshape(nb_edges,)))

# plt.show()






