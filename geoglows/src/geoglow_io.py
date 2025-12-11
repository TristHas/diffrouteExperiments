import numpy as np
import pandas as pd
import torch
import xarray as xr
import geopandas as gpd
import networkx as nx
from tqdm.notebook import tqdm
from pathlib import Path

from diffroute import read_params, get_node_idxs
from .config import DATA_ROOT

DATA_ROOT = Path("/data_prediction005/SYSTEM/prediction002/home/tristan/data/geoflow/data")

def extract_all_graphs(config_root, vpu_numbers):
    """
    """
    gs = []
    for vpu in tqdm(vpu_numbers):
        g, params = read_vpu_graph(config_root, vpu)
        for n in g.nodes:
            g.nodes[n]["x"] = params.loc[n, "x"]
            g.nodes[n]["k"] = params.loc[n, "k"]
        gs.append(g)
    return nx.compose_all(gs)

def read_vpu_graph(config_root, vpu):
    """
    """
    df = pd.read_csv(config_root / vpu / "rapid_connect.csv", header=None)
    g = nx.DiGraph()
    g.add_edges_from(df[[0,1]].values)
    g.remove_node(-1)

    idx = pd.read_csv(config_root / vpu / "riv_bas_id.csv", header=None).values.squeeze()
    k = pd.read_csv(config_root   / vpu / "k.csv", header=None).values.squeeze()
    x = pd.read_csv(config_root   / vpu / "x.csv", header=None).values.squeeze()
    params = pd.DataFrame({"k":k, "x":x}, index=idx)
    return g, params

def load_geoflow(root=DATA_ROOT, serialized_g=True, vpu_numbers=None):
    """
    """
    print("Loading runoffs...")
    df = pd.read_feather(root / "daily_sparse_runoff.feather")
    weight_df = pd.read_feather(root / "interp_weight.feather")

    config_root = root / 'configs'
    if vpu_numbers is None:
        vpu_numbers = [x.name for x in config_root.glob("*")]
    
    G = extract_all_graphs(config_root, vpu_numbers)
    for n in G.nodes: G.nodes[n]["k"] /= (3600*24)
    return G, df, weight_df

def annotate_graph_with_physical_prop(g, gdf):
    for n in g.nodes:
        g.nodes[n]["is_lake"] = (gdf.loc[n, "musk_x"]==.01)
        g.nodes[n]["dist"] = gdf.loc[n, "LengthGeodesicMeters"]
        g.nodes[n]["upa"] = np.sqrt(gdf.loc[n, "DSContArea"])

def read_physical_prop(g, nodes_idx=None, stds=None):
    nodes_idx = get_node_idxs(g) if nodes_idx is None else nodes_idx
    p_name = ["is_lake", "dist", "upa"]
    params = torch.tensor([[g.nodes[n][p] for p in p_name] for n in nodes_idx.index])
    if stds is None:
        stds = params.std(0, keepdims=True)
        stds[0,0]=1
    return params / stds, stds