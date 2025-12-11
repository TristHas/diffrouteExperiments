from pathlib import Path

import pandas as pd
import torch

from diffroute import get_node_idxs


class Dataset():
    def __init__(self, inp, lbl, g, device, pred_len=200, init_window=100, bs=None):
        """
        """
        node_idxs = get_node_idxs(g)
        self.inp, self.lbl, self.g = inp[node_idxs.index], lbl, g
        std = lbl.std().mean()
        self.X = torch.from_numpy(inp.loc[:"2016"].values).t().to(device).unsqueeze(-1)
        self.Y = torch.from_numpy(lbl.loc[:"2016"].values).t().to(device).unsqueeze(-1) / std
        self.Xte = torch.from_numpy(inp.loc["2017":].values).t().to(device).unsqueeze(-1)
        self.Yte = torch.from_numpy(lbl.loc["2017":].values).t().to(device).unsqueeze(-1) / std
        self.pred_len = pred_len
        self.init_window = init_window
        self.device = device
        self.std = std
        self.node_idxs = node_idxs
        self.bs = bs
        
    def sample(self):
        """
        """
        l = self.init_window + self.pred_len
        delta = self.X.shape[1] // l - 2
        
        start_idx = torch.randint(low=0,high=l, size=(1,))
        end_idx = start_idx + delta * l
        x = self.X[:,start_idx : end_idx].view(self.X.shape[0], delta, l, -1).permute(1,0,2,3)
        y = self.Y[:,start_idx : end_idx].view(self.Y.shape[0], delta, l).permute(1,0,2)
        if self.bs is not None:
            idxs = torch.randint(0, delta, (self.bs,))
            x = x[idxs]
            y = y[idxs]
        return x,y

    @classmethod
    def load_from_folder(cls, name, device, pred_len=200, init_window=100, **kwargs):
        inp, lbl, g = load_data(name)
        return cls(inp, lbl, g, device, pred_len=pred_len, init_window=init_window, **kwargs)

def load_data(name="40_P"):
    ds_root = Path("../data")
    ds_root = ds_root / name
    g = pd.read_pickle(ds_root / "g.pkl")
    inp = pd.read_pickle(ds_root / "inp.pkl")
    lbl = pd.read_pickle(ds_root / "lbl.pkl")
    
    return inp, lbl, g