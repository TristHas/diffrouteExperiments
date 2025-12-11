from tqdm import tqdm
import torch
import torch.nn as nn
from diffroute import LTIRouter, get_node_idxs, read_params
from .geoglow_io import read_physical_prop

class CalibratedRouting(nn.Module):
    def __init__(self, g, 
                 nodes_idx,
                 max_delay, 
                 block_size,
                 irf_fn, 
                 irf_agg, 
                 index_precomp, 
                 runoff_to_output,
                 dt, sampling_mode,
                param_mode="sigmoid"):
        super().__init__()
        self.model =  LTIRouter(g, 
                               nodes_idx=nodes_idx,
                               max_delay=max_delay, 
                               block_size=block_size,
                               irf_fn=irf_fn, 
                               irf_agg=irf_agg, 
                               index_precomp=index_precomp, 
                               runoff_to_output=runoff_to_output,
                               dt=dt, sampling_mode=sampling_mode,
                               )#.to(device)
        params = read_params(g, irf_fn, nodes_idx).float()
        self.register_buffer("offset", torch.tensor([.005, .0])[None])
        self.register_buffer("range", torch.tensor([.25, 1.2])[None])
        self.register_parameter("params", torch.nn.Parameter(params))
        self.param_mode = param_mode

    def init_params(self, init_x=.25, init_z=.2):
        with torch.no_grad():
            init = torch.zeros_like(self.params)
            init[:,1]-=3
            self.params[:]=init

    def forward(self, x):
        if self.param_mode=="sigmoid":
            params = torch.sigmoid(self.params) *self.range + self.offset
        else:
            params = self.params
        return self.model(x, params)

def read_physical_prop(g, nodes_idx=None, stds=None):
    nodes_idx = get_node_idxs(g) if nodes_idx is None else nodes_idx
    p_name = ["is_lake", "dist", "upa"]
    params = torch.tensor([[g.nodes[n][p] for p in p_name] for n in nodes_idx.index])
    if stds is None:
        stds = params.std(0, keepdims=True)
        stds[0,0]=1
    return params / stds, stds

class LearnedRouting(nn.Module):
    def __init__(self, g, 
                 nodes_idx,
                 max_delay, 
                 block_size=16,
                 irf_fn="muskingum", 
                 irf_agg="log_triton", 
                 index_precomp="cpu", 
                 runoff_to_output=False,
                 dt=1/24, sampling_mode="avg",
                 mlp=None, param_stds=None):
        super().__init__()
        self.model =  LTIRouter(g, 
                               nodes_idx=nodes_idx,
                               max_delay=max_delay, 
                               block_size=block_size,
                               irf_fn=irf_fn, 
                               irf_agg=irf_agg, 
                               index_precomp=index_precomp, 
                               runoff_to_output=runoff_to_output,
                               dt=dt, sampling_mode=sampling_mode,
                               )#.to(device)
        
        self.register_buffer("offset", torch.tensor([.005, .0])[None])
        self.register_buffer("pre_sigmoid_offset", torch.tensor([.0, 3.])[None])
        self.register_buffer("range", torch.tensor([.25, 1.2])[None])
        self.register_buffer("params", read_physical_prop(g, nodes_idx, stds=param_stds)[0].float())
        if mlp is None:
            self.mlp = MLP(in_dim=self.params.shape[-1], out_dim=2, n_layers=3, hidden_dim=256)
        else:
            self.mlp = mlp
            
    def forward(self, x):
        params = self.mlp(self.params[None]).squeeze()
        params = torch.sigmoid(params - self.pre_sigmoid_offset) *self.range + self.offset
        return self.model(x, params)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers=3, hidden_dim=256):
        super().__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        # Hidden layers
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Output layer
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
        
class MultiStageLearnedMLP(nn.Module):
    def __init__(self, clusters_g, node_transfer, cat,
                 time_window=50, dt=1/24,
                 model_name="muskingum",
                 param_stds=None):
        """
        """
        super().__init__()
        self.mlp = MLP(in_dim=3, out_dim=2, n_layers=3, hidden_dim=256)
        routers, params, node_idxs = [],[],[]
        for i,g in tqdm(enumerate(tqdm(clusters_g))):
            nodes_idx = get_node_idxs(g)
            model = LearnedRouting(g, 
                                   irf_fn=model_name,
                                   nodes_idx=nodes_idx,
                                   max_delay=time_window, 
                                   runoff_to_output=False,
                                   dt=dt, mlp=self.mlp,
                                   param_stds=param_stds,
                                   )
            routers.append(model)
            node_idxs.append(nodes_idx)
            
        self.routers = nn.ModuleList(routers)
        self.node_idxs = node_idxs
        self.cat = cat
        self.node_transfer = node_transfer
            
    def init_layers_buffers(self, device):
        """
        """
        for router in self.routers: router.model.aggregator.init_buffers(device)

    def forward(self, N=None):
        """
        """
        if N is None: N = len(self.routers)
        outputs = []
        transfered_inputs = {i:[] for i,_ in enumerate(self.routers)}

        for i in range(N):
            model = self.routers[i]
            #params = self.params[i]
            x = self.cat.read_catchment(i).clone()  / (3600 * 24)
            
            for e_dst, inp_dis in transfered_inputs[i]: x[:,e_dst] += inp_dis.squeeze()
            out = model(x)
            for (c_idx, e_src, e_dst) in self.node_transfer[i]: transfered_inputs[c_idx].append((e_dst, out[:,e_src].clone().detach()))   
                
            outputs.append(out)
        return outputs