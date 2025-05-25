import networkx as nx
import sys

def generate_smallworld_graph(n_nodes):
    # Watts-Strogatz model with k=100 neighbors and rewiring prob p=0.1
    return nx.watts_strogatz_graph(n_nodes, 100, 0.1).to_directed()

def generate_scalefree_graph(n_nodes):
    # Barabási–Albert model with m=5 new edges per node
    return nx.barabasi_albert_graph(n_nodes, 5).to_directed()

def generate_random_graph(n_nodes):
    # Erdős–Rényi model with p=0.01 edge creation probability
    return nx.erdos_renyi_graph(n_nodes, 0.01).to_directed()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python graph_generator.py <num_nodes> <graph_type>")
        print("       graph_type = smallworld | scalefree | random")
        sys.exit(1)

    num_nodes = int(sys.argv[1])
    graph_type = sys.argv[2].lower()

    if graph_type == "smallworld":
        G = generate_smallworld_graph(num_nodes)
    elif graph_type == "scalefree":
        G = generate_scalefree_graph(num_nodes)
    elif graph_type == "random":
        G = generate_random_graph(num_nodes)
    else:
        print("Invalid graph_type. Choose: smallworld | scalefree | random")
        sys.exit(1)

    if G.number_of_edges() == 0:
        print("Generated graph has no edges.", file=sys.stderr)
        sys.exit(2)
    nx.write_edgelist(G, sys.stdout.buffer, delimiter="\t", data=False)
