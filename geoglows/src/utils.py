def nse_loss_fn(out, y):
    """
    """
    mse = (out - y)**2
    nse = mse / y.std(-1, keepdims=True)
    return nse.mean()