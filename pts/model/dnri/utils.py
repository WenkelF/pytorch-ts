import numpy as np


def construct_full_graph(target_dim):
    adj = np.ones(target_dim) - np.eye(target_dim)
    send_edges = np.where(adj)[0]
    recv_edges = np.where(adj)[1]
    
    return send_edges, recv_edges


def construct_expander(target_dim, density):
    num_edges_full = (target_dim ** 2 - target_dim)/2
    num_edges_target = int(np.floor(density * num_edges_full))
    
    # split nodes in two disjoint sets
    m = int(np.floor(target_dim/2))
    nodes_perm = np.random.permutation(target_dim)
    nodes_base = nodes_perm[:m]
    nodes_rest = nodes_perm[m:]
    
    # number of iterations for regular bipartite graph of (at most) desired density
    num_iter = int(np.floor(num_edges_target/m))
    print("Number of iterations: "+str(num_iter))
    print("Number of edges: "+str(num_iter*m))
    print("Desired density: "+str(density))
    print("Actual density: "+str(np.round(num_iter*m/num_edges_full, 3)))
    
    send_edges = np.tile(nodes_base, num_iter)
    recv_edges_list = [nodes_rest[:m]]
    num_tries_list = []
    
    for _ in range(num_iter-1):
        num_dupl = 1
        num_tries = 0
        
        # check for duplicates
        while num_dupl > 0:
            # permute reciever nodes
            nodes_temp = np.random.permutation(nodes_rest)[:m]
            num_dupl = sum([sum(recv_edges_list[i] == nodes_temp) for i in range(len(recv_edges_list))])
            num_tries += 1
        recv_edges_list.append(nodes_temp)
        num_tries_list.append(num_tries)

    print(num_tries_list)
    
    recv_edges = np.concatenate(recv_edges_list)
    
    # make edges bi-directional
    return np.concatenate((send_edges, recv_edges), axis=0), np.concatenate((recv_edges, send_edges), axis=0)


def construct_expander_fast(target_dim, density):
    num_edges_full = (target_dim ** 2 - target_dim)/2
    num_edges_target = int(np.floor(density * num_edges_full))
    
    m = int(np.floor(target_dim/2))
    nodes_perm = np.random.permutation(target_dim)
    nodes_base = nodes_perm[:m].reshape(-1,1)
    nodes_rest = nodes_perm[m:].reshape(-1,1)
    
    edges = np.concatenate((nodes_base, nodes_rest[:m]), axis=1)
    num_edges=len(edges)
    
    num_iter = 0
    while num_edges < num_edges_target:
        nodes_temp = np.random.permutation(nodes_rest)[:m]
        
        edges = np.concatenate((edges, np.concatenate((nodes_base, nodes_temp), axis=1)), axis=0)
        # delete duplicates
        np.unique(edges, axis=0)

        num_edges = len(edges)
        num_iter += 1

    # prune to target number of edges
    edges = edges[:num_edges_target,:]
    # make edges bi-directional
    edges = np.concatenate((edges, edges[:,[1,0]]), axis=0)

    print("Number of iterations: "+str(num_iter))
    print("Number of edges: "+str(int(len(edges)/2)))
    print("Desired density: "+str(density))
    print("Actual density: "+str(np.round(len(edges)/2/num_edges_full, 3)))
    
    return edges[0,:].squeeze(), edges[1,:].squeeze()


def construct_random_graph(target_dim, density):
    num_edges_full = (target_dim ** 2 - target_dim)/2
    num_edges_target = int(np.floor(density * num_edges_full))
    
    all_edges = np.where(np.triu(np.ones(target_dim) - np.eye(target_dim)))
    
    all_send_edges = all_edges[0]
    all_recv_edges = all_edges[1]
    
    reorder = np.random.permutation(len(all_send_edges))
    
    send_edges = all_send_edges[reorder][:num_edges_target]
    recv_edges = all_recv_edges[reorder][:num_edges_target]
    
    print("Number of edges: "+str(int(len(send_edges))))
    print("Desired density: "+str(density))
    print("Actual density: "+str(np.round(len(send_edges)/num_edges_full, 3)))
    
    # make edges bi-directional
    return np.concatenate((send_edges, recv_edges), axis=0), np.concatenate((recv_edges, send_edges), axis=0)


def construct_bipartite_graph(target_dim, density):
    num_edges_full = (target_dim ** 2 - target_dim)/2
    
    num_nodes = int(np.floor((target_dim - np.sqrt(target_dim ** 2 - 4 * density * num_edges_full)) / 2))
    
    nodes_perm = list(np.random.permutation(target_dim))
    nodes_base = nodes_perm[:num_nodes]
    nodes_rest = nodes_perm[num_nodes:]
    
    m = len(nodes_rest)
    
    send_edges = []
    recv_edges = []
    
    for node in nodes_base:
        node_stacked = m*[node]
    
        send_edges += node_stacked
        recv_edges += nodes_rest
        # make edges bi-directional
        send_edges += nodes_rest
        recv_edges += node_stacked
    
    print("Number of edges: "+str(int(len(send_edges)/2)))
    print("Number of super-nodes: "+str(num_nodes))
    print("Desired density: "+str(density))
    print("Actual density: "+str(np.round(len(send_edges)/2/num_edges_full, 3)))
    
    return np.array(send_edges), np.array(recv_edges)


# old functions --------------------------------------------------

def construct_expander_old(target_dim, density):
    num_edges_full = (target_dim ** 2 - target_dim)/2
    num_edges = density * num_edges_full
    
    m = int(np.floor(target_dim/2))
    nodes_perm = list(np.random.permutation(target_dim))
    nodes_base = nodes_perm[:m]
    nodes_rest = nodes_perm[m:]
    
    num_iter = int(np.floor(num_edges / m))
    print("Approx. number of edges: "+str(num_iter*m))
    print("Desired density: "+str(density))
    print("Approx. density: "+str(num_iter*m/num_edges_full))
    
    send_edges = []
    recv_edges = []
    
    for _ in range(num_iter):
        nodes_temp = list(np.random.permutation(nodes_rest)[:m])
        send_edges += nodes_base
        recv_edges += nodes_temp
        send_edges += nodes_temp
        recv_edges += nodes_base
    
    return np.array(send_edges), np.array(recv_edges)


def construct_random_graph_old(target_dim, density):
    total_num_edges = (target_dim ** 2 - target_dim)/2
    num_edges = int(np.floor(density * total_num_edges))
    
    nodes_base = list(np.random.choice(target_dim, num_edges))
    nodes_temp = list(np.random.choice(target_dim, num_edges))
    
    send_edges = []
    recv_edges = []
    
    send_edges += nodes_base
    recv_edges += nodes_temp
    send_edges += nodes_temp
    recv_edges += nodes_base
    
    print("Approx. number of edges: "+str(len(send_edges)/2))
    print("Desired density: "+str(density))
    print("Approx. density: "+str(len(send_edges)/2/total_num_edges))
    
    return np.array(send_edges), np.array(recv_edges)


def construct_bipartite_graph_old(target_dim, density):
    total_num_edges = (target_dim ** 2 - target_dim)/2
    
    num_nodes = int(np.floor(density * total_num_edges / target_dim))
    
    nodes_perm = list(np.random.permutation(target_dim))
    nodes_base = nodes_perm[:num_nodes]
    nodes_rest = nodes_perm[num_nodes:]
    
    m = len(nodes_rest)
    
    send_edges = []
    recv_edges = []
    
    for node in nodes_base:
        node_stacked = m*[node]
    
        send_edges += node_stacked
        recv_edges += nodes_rest
        send_edges += nodes_rest
        recv_edges += node_stacked
    
    print("Approx. number of edges: "+str(len(send_edges)/2))
    print("Number of super-nodes: "+str(num_nodes))
    print("Desired density: "+str(density))
    print("Approx. density: "+str(len(send_edges)/2/total_num_edges))
    
    return np.array(send_edges), np.array(recv_edges)
    