import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm

from diffroute import get_node_idxs

class CatchmentInterpolator:
    def __init__(self, gs, runoff, weight_df, device="cpu"):
        """
        Parameters:
          gs: list of subgraphs
          runoff: DataFrame of runoff (rows: time, columns: pixels)
          weight_df: DataFrame with columns: 'pixel_idx', 'river_id', 'area_sqm_total'
          device: device to store tensors (e.g. "cpu" or "cuda")
        """
        self.device = device
        # Map runoff column labels to their numeric indices.
        self.map_inp = pd.Series(np.arange(len(runoff.columns)), index=runoff.columns)
        # Convert runoff data to a torch.Tensor on the desired device.
        self.runoff = torch.from_numpy(runoff.values).to(device)
        # Store subgraphs in a dictionary with an integer key.
        self.gs = {i: g for i, g in enumerate(gs)}
        # Set the weight_df index to be river_id for fast lookup.
        self.weight_df = weight_df.sort_values("river_id").set_index("river_id")
        # This will hold precomputed indices for each subgraph.
        self.init_all_indices()

    def init_all_indices(self):
        """
        Precompute interpolation indices for all subgraphs.
        """
        self.indices = {k: self.init_indices(k) for k in tqdm(self.gs)}

    def init_indices(self, idx):
        """
        For a given subgraph, precompute the interpolation indices.
        
        Returns:
          n_cats: int, the number of catchments (output columns) in the subgraph.
          src_idxs: 1D torch.Tensor containing indices into runoff (source columns).
          dest_idxs: 1D torch.Tensor containing destination indices for scatter-add.
          weights: 1D torch.Tensor of weights (area_sqm_total) corresponding to each pixel.
        
        Assumes get_node_idxs(g) returns a pandas Series whose index are the river_ids and whose values
        provide the output order for each catchment.
        """
        g = self.gs[idx]
        map_out = get_node_idxs(g)  # expected to be a pandas Series
        n_cats = len(map_out)
        # Re-index weight_df to follow the order of catchment nodes in map_out.
        weight_subset = self.weight_df.loc[map_out.index]
        # Destination indices: using the order provided by map_out.
        dest_idxs = torch.tensor(map_out[weight_subset.index].values, 
                                 dtype=torch.long, device=self.device)
        # Source indices: for each entry in weight_subset, convert the 'pixel_idx'
        # (which should match a column label in runoff) to its numeric index.
        src_idxs = torch.tensor(self.map_inp.loc[weight_subset["pixel_idx"]].values,
                                dtype=torch.long, device=self.device)
        # Weights: the area_sqm_total values as a float tensor.
        weights = torch.tensor(weight_subset["area_sqm_total"].values,
                               dtype=torch.float, device=self.device)
        return n_cats, src_idxs, dest_idxs, weights

    def read_catchment(self, idx):
        """
        Apply the interpolation for a given subgraph.
        
        Returns:
          out: torch.Tensor of shape (T, n_cats) containing the aggregated (weighted) runoff.
        """
        # Retrieve precomputed indices.
        n_cats, src_idxs, dest_idxs, weights = self.indices[idx]
        # Gather the runoff data for the required pixels (columns).
        # Note: runoff is of shape (T, num_pixels), so we select along dim 1.
        x = self.runoff[:, src_idxs]  # shape: (T, K)
        # Multiply each selected column by its weight.
        weighted = x * weights  # broadcasts over the time dimension
        # Create an output tensor to accumulate the weighted sums.
        out = torch.zeros(x.shape[0], n_cats, dtype=x.dtype, device=self.device)
        # Scatter-add: for each column in weighted, add it to the corresponding catchment column in out.
        out.index_add_(1, dest_idxs, weighted)
        return out.t()[None]

    def __getitem__(self, idx):
        return self.read_catchment(idx)
    
    def __iter__(self):
        for i in range(len(self)):
            return self[i]

    def __len__(self):
        return len(self.self.gs)
        

def process(df, weight_df):
    df2 = df[weight_df["pixel_idx"]]
    df2.columns = weight_df['river_id']
    df2 = df2 * weight_df['area_sqm_total'].values
    df2 = df2.T.groupby(by=df2.columns).sum().T
    #time_diff = np.diff(df2.index)
    #assert (time_diff.astype(int)==86400000000000).all()
    return df2
    
def test_interpolation(clusters_g, runoff, weight_df):
    catch_interp = CatchmentInterpolator(clusters_g, runoff, weight_df)
    catch_interp.init_all_indices()
    # Loop over each subgraph (cluster) and compare the outputs of process() vs CatchmentInterpolator
    for idx, g in enumerate(clusters_g):
        # Get catchment labels (assumed to be the index of the Series returned by get_node_idxs)
        node_idxs = get_node_idxs(g)  # e.g., Series with index=river_ids and values=dest order
        columns = node_idxs.index
        
        # Compute interpolation using the original process() function.
        # Note: weight_df_subset selects only the rows corresponding to the current catchment.
        weight_df_subset = weight_df[weight_df.river_id.isin(columns)]
        df_process = process(runoff, weight_df_subset)
        # Ensure that the columns are in the same order as in node_idxs.
        df_process = df_process[columns]
        
        # Compute interpolation using the GPU-based (torch) method.
        torch_out = catch_interp.read_catchment(idx)
        # Convert torch tensor to DataFrame (using the same index and column order).
        df_torch = pd.DataFrame(torch_out.cpu().numpy(), index=runoff.index, columns=columns)
        
        # Compare the two results.
        diff = np.abs(df_process.values - df_torch.values)
        max_diff = diff.max()
        print(f"Cluster {idx}: max difference = {max_diff}")
        
        # Assert that the outputs are close.
        np.testing.assert_allclose(df_process.values, df_torch.values, rtol=1e-5)