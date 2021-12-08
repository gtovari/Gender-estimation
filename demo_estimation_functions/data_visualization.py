"""Visualization function examples for the homework project"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


def plot_degree_distribution(G):
    ##Plot a degree distribution of a graph
    plot_df = (
        pd.Series(dict(G.degree)).value_counts().sort_index().to_frame().reset_index()
    )
    plot_df.columns = ["k", "count"]
    plot_df["log_k"] = np.log(plot_df["k"])
    plot_df["log_count"] = np.log(plot_df["count"])
    fig, ax = plt.subplots()

    ax.scatter(plot_df["k"], plot_df["count"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.suptitle("Mutual Degree Distribution")
    ax.set_xlabel("k")
    ax.set_ylabel("count_k")
    
def plot_age_distribution_by_gender(nodes):
    """Plot a histogram where the color represents gender"""
    plot_df = nodes[["AGE", "gender"]].copy(deep=True).astype(float)
    plot_df["gender"] = plot_df["gender"].replace({0.0: "woman", 1.0: "man"})
    sns.histplot(data=plot_df, x="AGE", hue="gender", bins=np.arange(0, 45, 5) + 15)

########################################################################################################################  
def plot_degree_by_gender(nodes, G):
    
    ##Plot degree centrality by age where colors represent each gender
    
    nodes_w_degree = nodes.set_index("user_id").merge(
        pd.Series(dict(G.degree)).to_frame(),
        how="left",
        left_index=True,
        right_index=True,
    )
    nodes_w_degree = nodes_w_degree.rename({0: "degree"}, axis=1)
    plot_df = (
        nodes_w_degree.groupby(["AGE", "gender"]).agg({"degree": "mean"}).reset_index()
    )
    sns.lineplot(data=plot_df, x="AGE", y="degree", hue="gender").set_title("Degree Centrality")
    plt.legend(title = "Gender", labels = ["Male", "Female"])
    
    
def plot_neighbor_by_gender(nodes, G):
    
    ## Plot neighbor connectivity by age where colors represent each gender
    
    nodes_neighbor = nodes.assign(neighbor_connectivity = nodes["user_id"].map(nx.average_neighbor_degree(G)))
    plot_df = (nodes_neighbor.groupby(["AGE", "gender"]).agg({"neighbor_connectivity": "mean"}).reset_index())
    sns.lineplot(data=plot_df, x="AGE", y="neighbor_connectivity", hue="gender").set_title("Neighbor Connectivity")
    plt.legend(title = "Gender", labels = ["Male", "Female"])
    
def plot_clustering_by_gender(nodes, G):
    
     ## Plot local clustering coefficient (cc) by age where colors represent each gender
    
    clustering = nodes.assign(clustering_coef = nodes["user_id"].map(nx.clustering(G)))
    plot_df = (clustering.groupby(["AGE", "gender"]).agg({"clustering_coef": "mean"}).reset_index())
    sns.lineplot(data=plot_df, x="AGE", y="clustering_coef", hue="gender").set_title("Triadic Closure")
    plt.legend(title = "Gender", labels = ["Male", "Female"])
    
def embeddedness_by_gender(nodes, G):
    
    #calculating embedddness by the given formula of the article (by the intersections and unions)
    
    embedd = []
    for user in nodes["user_id"]:
        neighbors = set(G.neighbors(user))
        N_u = len(set(G.neighbors(user)))
        sum = 0
        for neighbor in neighbors:
            intersection = len(neighbors.intersection(set(G.neighbors(neighbor))))
            union = len(neighbors.union(set(G.neighbors(neighbor))))
            sum += intersection / union
        total = sum / N_u
        embedd.append(total)
        
    nodes_w_embeddedness = nodes.assign(embeddedness = embedd)
    return nodes_w_embeddedness
    
def plot_embeddedness_by_gender(nodes, G):
    
    ## Plot embeddedness by age where colors represent each gender
    
    nodes_w_embeddedness = embeddedness_by_gender(nodes, G)
    plot_df = (nodes_w_embeddedness.groupby(["AGE", "gender"]).agg({"embeddedness": "mean"}).reset_index())
    sns.lineplot(data=plot_df, x="AGE", y="embeddedness", hue="gender").set_title('Embeddedness')
    plt.legend(title = "Gender", labels = ["Male", "Female"])

#####################################################################################################################
    

def plot_age_relations_heatmap(edges_w_features):
    """Plot a heatmap that represents the distribution of edges"""
    # TODO: check what happpens without logging
    # TODO: instead of logging check what happens if you normalize with the row sum
    #  make sure you figure out an interpretation of that as well!
    # TODO: separate these charts by gender as well
    # TODO: column names could be nicer
    plot_df = edges_w_features.groupby(["gender_x", "gender_y", "AGE_x", "AGE_y"]).agg(
        {"smaller_id": "count"}
    )
    plot_df_w_w = plot_df.loc[(0, 0)].reset_index()
    plot_df_heatmap = plot_df_w_w.pivot_table(
        index="AGE_x", columns="AGE_y", values="smaller_id"
    ).fillna(0)
    plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
    sns.heatmap(plot_df_heatmap_logged)
