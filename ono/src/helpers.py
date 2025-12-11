from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

def hydro_metrics(O: pd.Series, Y: pd.Series) -> pd.Series:
    """
    """
    def kge(r_, a_, b_):
        if np.isnan(r_) or np.isnan(a_) or np.isnan(b_): return np.nan
        return 1.0 - np.sqrt((r_ - 1.0)**2 + (a_ - 1.0)**2 + (b_ - 1.0)**2)

    O, Y = O.astype(float).align(Y.astype(float), join="inner")
    mask = O.notna() & Y.notna()
    O, Y = O[mask], Y[mask]
    if len(O) == 0: return pd.Series(dtype=float)

    # Basic stats
    mu_o, mu_y = O.mean(), Y.mean()
    std_o, std_y = O.std(ddof=1), Y.std(ddof=1)
    var_y = Y.var(ddof=1)
    mse = ((Y - O) ** 2).mean()
    mae = (Y - O).abs().mean()
    r = O.corr(Y)

    # Components
    beta  = np.nan if mu_y == 0 else (mu_o / mu_y)
    alpha = np.nan if std_y == 0 else (std_o / std_y)
    cv_o  = np.nan if mu_o == 0 else (std_o / mu_o)
    cv_y  = np.nan if mu_y == 0 else (std_y / mu_y)
    gamma = np.nan if (cv_y == 0 or np.isnan(cv_y)) else (cv_o / cv_y)

    # Scores
    nse = np.nan if var_y == 0 else 1.0 - mse / var_y
    kge_2009 = kge(r, alpha, beta)   # r, α, β
    kge_2012 = kge(r, gamma, beta)   # r, γ, β

    return pd.Series({
        "NSE": nse,
        "KGE_2009": kge_2009,
        "KGE_2012": kge_2012,
        "r": r,
        "alpha": alpha,   
        "gamma": gamma,   
        "beta": beta,    
        "MAE": mae,
        "MSE": mse,
        "n": len(O),
    })

def training_loop(ds, model, opt, 
                  n_iter=100, 
                  n_epoch=80, 
                  init_window=100, 
                  scheduler=None,
                  clip_grad_norm=None):
    """
        Training loop with an optional learning rate scheduler.
    """
    losses, nses, trnses = [],[],[]

    for epoch in range(1, n_epoch):
        for i in tqdm(range(n_iter)):
            x, y = ds.sample()
            
            o = model(x)
            o = o[..., init_window:]
            y = y[..., init_window:]
            
            loss = ((y[:, 0] - o)**2).mean()
            
            opt.zero_grad()
            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                               max_norm=clip_grad_norm)
            opt.step()
            losses.append(loss.item())

        if scheduler is not None: scheduler.step()

    return pd.Series(losses)

def extract_train(model, ds, init_window):
    with torch.no_grad(): O = model(ds.X[None,:,:65536-1])
    YY = pd.Series(ds.Y[0].cpu().squeeze()).iloc[init_window:]
    OO = pd.Series(O.squeeze().detach().cpu()).iloc[init_window:]
    return YY, OO

def extract_test(model, ds, init_window):
    with torch.no_grad(): O = model(ds.Xte[None])
    YY = pd.Series(ds.Yte[0].cpu().squeeze()).iloc[init_window:]
    OO = pd.Series(O.squeeze().detach().cpu()).iloc[init_window:]
    return YY, OO
