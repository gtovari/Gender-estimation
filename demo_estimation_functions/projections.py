"""Functions to support gender projections"""
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import networkx as nx
from sklearn.metrics import accuracy_score, confusion_matrix

#####################################################################################################################

def prediction_from_neighbors(G, nodes, test_df):
    
    # This function predicts each users' gender by calculating the number of male and female connections,
    # and given the basic assumptation that male have more male, and female have more female friends,
    # guess the gender of the tested dataset's observations.   
    
   
    prediction = []
    # iterating through all observations in the test dataset and counting the number of male and female connections
    for users in test_df["user_id"]:
        # listing neighbors
        neighbors = set(G.neighbors(users))
        
    # filtering the test_df dataframe for the neighbors and grouping the filtered one by gender and counting them. If the neighbor
    # has na as gender info, they are dropped from the calculation.
        
        genders = nodes[nodes.user_id.isin(neighbors)].dropna().groupby("gender").count()
        
        try:
            males = genders.loc[1, 'user_id']
        except:
            males = 0
        try:
            females = genders.loc[0, 'user_id']
        except:
            females = 0

        if males >= females:
            prediction.append(1)
        else:
            prediction.append(0)
    
    # putting together the dataset containing the predictions based on this basic model

    prediction_df = pd.DataFrame(dict(user_id = test_df.user_id, gender = prediction))

    return prediction_df

################################################################################################################
