import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from diffroute import LTIRouter, get_node_idxs

# This is an important part that we should take more care in making easy to use.
# We should either include it in the diffroute repo or make a separate repo with all the helpers needed.

PARAMS_BOUNDS = {
    "muskingum":(torch.tensor([.1, 0.01])[None], # Parameter k and v minimum values
                 torch.tensor([5, .4])[None],    # Parameter k and v values range
                 torch.tensor([-3., -3.])[None]),# Parameter k and v initial values (before sigmoid)
    "pure_lag":(torch.tensor([.01])[None],       # Parameter lag minimum value
                torch.tensor([5.])[None],        # Parameter lag value range
                torch.tensor([-3.])[None]),      # Parameter initial values (before sigmoid)
    "linear_storage":(torch.tensor([.1])[None],  
                      torch.tensor([9.9])[None],
                      torch.tensor([0.])[None]),
    'nash_cascade':(torch.tensor([.05])[None], 
                    torch.tensor([3.25])[None],
                    torch.tensor([-3.])[None]),
    'hayami':(torch.tensor([.2,   .1])[None], 
              torch.tensor([.8,  5.9])[None],
              torch.tensor([-3,  -3.])[None]),
}

MS_TO_MMKM2 = 10**12 / (24 * 3600 * 10**9)
RUNOFF_STD  = 12

def init_model(g, inp_size, irf_fn, device, 
               dt=1, cascade=1):
    """
    """
    channel_feats = pd.read_pickle("../data/40_P/river_static.pkl")
    channel_feats["channel_length"] = channel_feats["channel_length"].clip(.01)
    model = HydroModel(g, irf_fn=irf_fn,
                       inp_size=inp_size,
                       node_idxs=get_node_idxs(g),
                       channel_feats=channel_feats,
                       runoff_to_output=False,
                       dt=dt, cascade=cascade).to(device)
    return model

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=1, num_layers=1, softplus=True):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.out_act = nn.Softplus() if softplus else nn.Identity()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.linear(lstm_out)
        return self.out_act(out.squeeze())
        
class HydroModel(nn.Module):
    def __init__(self, g, inp_size, 
                 node_idxs, channel_feats, irf_fn,
                 lumped=False,
                 runoff_to_output=True,
                 max_delay=100,
                 dt=1,
                 cascade=1):
        """
        """
        super().__init__()
        assert irf_fn in PARAMS_BOUNDS, f"'{irf_fn}' is not a valid routing model name"
        n_params = PARAMS_BOUNDS[irf_fn][0].shape[-1]
        # Precompute initial buffers
        runoff_coefficient = torch.from_numpy(np.array([g.nodes[x]["catchment_area"] \
                                                        for x in node_idxs.index])) \
                             * MS_TO_MMKM2 * RUNOFF_STD
        static_feats = torch.from_numpy(channel_feats.loc[node_idxs.index].fillna(0.001).values)
        # Init buffers
        self.register_buffer("runoff_conversion", runoff_coefficient.float().unsqueeze(-1))
        self.register_buffer("static_feats", static_feats.float())
        self.register_buffer("out_min", PARAMS_BOUNDS[irf_fn][0])
        self.register_buffer("out_max", PARAMS_BOUNDS[irf_fn][1])
        self.register_buffer("pre_sig", PARAMS_BOUNDS[irf_fn][2])
        self.register_parameter("routing_params", nn.Parameter(torch.zeros((len(node_idxs), n_params)).float()))
        self.irf_fn = irf_fn
        self.lumped = lumped
        # Init modules
        self.runoff_model   = LSTMModel(input_size=inp_size)
        self.routing_model  = LTIRouter(g, 
                                        runoff_to_output=runoff_to_output,
                                        irf_fn=irf_fn,
                                        nodes_idx=node_idxs, 
                                        max_delay=max_delay,
                                        dt=dt, cascade=cascade,
                                        block_size=16)

    def read_params(self):
        """
        """
        params = torch.sigmoid(self.routing_params + self.pre_sig)
        params = self.out_min + (params * self.out_max)
        if self.irf_fn == "hayami": 
            # In hayami, do not learn the distance, it does not make sense, just use the true chanel distance.
            # Instead, only learn celerity and diffusion.
            # Other models do not explicitly make use of distance in their formula so we learn their implicit parameters
            params = torch.cat([self.static_feats[:,1].unsqueeze(-1), params], dim=-1)
        return params

    def forward(self, x):
        """
        """
        if self.lumped:
            return self.forward_lumped(x)
        else:
            return self.forward_structured(x)
    
    def forward_structured(self, x):
        """
        """
        params = self.read_params()
        runoffs = self.runoff_model(x.reshape(-1, x.shape[2], x.shape[3])).view(x.shape[:-1])
        runoffs = runoffs * self.runoff_conversion[None]
        discharges = self.routing_model(runoffs, params)
        discharges = discharges[:,2] / 37 # Normalization to equalize with the "averaged" meaning of the lumped model. 
        # It would be cleaner if we worked on sums instead, but there needs to be some normalization somewhere.
        # This is only done to get the equivalence between lumped and structured and is not necessary usually.
        return discharges
        
    def forward_lumped(self, x):
        """
        """
        rc = self.runoff_conversion.unsqueeze(0).unsqueeze(-1) /\
             self.runoff_conversion.sum()
        x = (x*rc).sum(1, keepdims=True) # Aggregate by weighted sum.
        runoffs = self.runoff_model(x.reshape(-1, x.shape[2], x.shape[3])).view(x.shape[:-1]).squeeze()
        discharges = runoffs * self.runoff_conversion.mean()
        return discharges