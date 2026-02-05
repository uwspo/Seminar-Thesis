import torch
import torch.nn as nn
from typing import Tuple, List
import matplotlib.pyplot as plt

#Used to reduce complexity
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Utility function that flattens a given tensor. 
"""
def _flatten(tlist: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tlist])
    
"""
Utility function that calculates the euclidean and maximum norm of a tensor.
"""
def _grad_norms(grads: List[torch.Tensor]) -> Tuple[float, float]:
    with torch.no_grad():
        gflat = _flatten([g.detach() for g in grads])
        return gflat.norm().item(), gflat.abs().max().item()

"""
"""
def forward_with_θ(model, X, θ):
    idx = 0
    out = X
    for module in model.net:
        if isinstance(module, nn.Linear):
            W = θ[idx]
            bias = θ[idx + 1]
            out = out @ W.T + bias
            idx += 2
        else:
            out = module(out)
    return out

"""
Generalized loss function that can be specified with multiple loss functions (such as mse or rss).
Calculates the forecast error for a specific model and a dataset.
"""
def loss_function(model, θ, X, Y, loss):
    Y_hat = forward_with_θ(model, X, θ)
    return float(loss(Y_hat, Y).detach())

"""
Calculates the loss function grad. The loss can be specified with multiple loss functions (such as mse or rss).
"""
def loss_function_grad(model, θ, X, Y, loss):
    θ_diff = [p.clone().detach().requires_grad_(True) for p in θ]
    Y_hat = forward_with_θ(model, X, θ_diff)
    loss_val = loss(Y_hat, Y)
    loss_val.backward()
    grads = [p.grad for p in θ_diff]
    return grads

"""
Implementation of the stochastic gradient descent algorithm.

model: Neural Network Object
θ0: Tensor Object for the Starting Configuration of the weights and biases
training_dataset: List of Tensor data points (x,y)
loss: loss function, can be either mse oder rss
η,b: Scalars
"""
def stochastic_gradient_descent(model, θ0, training_dataset, loss, η, b, plot=True):
    #Use device for reduced complexity
    model.to(device)

    # move initial params to device
    θ = [p.to(device).detach().clone() for p in θ0]

    #Initialize the history variables for specific information about the algorithm.
    θ_history, grad_history = [], []
    θ_norm_hist, step_abs_hist, step_rel_hist = [], [], []
    loss_before_hist, loss_after_hist, loss_delta_hist = [], [], []
    grad_l2_hist, grad_linf_hist = [], []
    lr_hist = []

    # Shuffle and minibatches
    data_shuffled = training_dataset[:]
    torch.random.manual_seed(torch.randint(0, 1<<31, (1,)).item())
    import random; random.shuffle(data_shuffled)

    batch_samples = [data_shuffled[i:i+b] for i in range(0, len(training_dataset), b)]

    current_lr = η

    #split training dataset into x and y values
    X_full = torch.stack([x for x, y in training_dataset]).to(device)
    Y_full = torch.stack([y for x, y in training_dataset]).to(device)

    for batch in batch_samples:
        X_batch = torch.stack([x for x, y in batch]).to(device)
        Y_batch = torch.stack([y for x, y in batch]).to(device)

        # loss before
        with torch.no_grad():
            L_before = loss_function(model, θ, X_full, Y_full, loss)

        # Gradienten berechnen
        grads = loss_function_grad(model, θ, X_batch, Y_batch, loss)
        g_l2, g_linf = _grad_norms(grads)

        # Parameter update
        θ_prev = [p.detach().clone() for p in θ]
        with torch.no_grad():
            θ = [p - current_lr * g for p, g in zip(θ, grads)]

        # Step sizes
        with torch.no_grad():
            delta = _flatten([c - p for p, c in zip(θ_prev, θ)])
            step_abs = delta.norm().item()
            θn = _flatten(θ).norm().item()
            step_rel = step_abs / (θn + 1e-12)

        # Loss after
        with torch.no_grad():
            L_after = loss_function(model, θ, X_full, Y_full, loss)

        # Log history
        θ_history.append([p.detach().clone() for p in θ])
        grad_history.append([g.detach().clone() for g in grads])

        θ_norm_hist.append(θn)
        step_abs_hist.append(step_abs)
        step_rel_hist.append(step_rel)
        loss_before_hist.append(L_before)
        loss_after_hist.append(L_after)
        loss_delta_hist.append(L_after - L_before)
        grad_l2_hist.append(g_l2)
        grad_linf_hist.append(g_linf)
        lr_hist.append(current_lr)

    #Replace the new θ of the neural network.
    model.replace_θ([p.detach().cpu() for p in θ])

    #Plots for useful information about the training process
    if plot:
        plt.figure()
        plt.plot(step_abs_hist, color="blue")
        plt.title("Abstand der SGD-Iterate ‖θ_{t+1}-θ_t‖")
        plt.xlabel("Iteration"); plt.ylabel("absolute step"); plt.show()

        plt.figure()
        plt.plot(step_rel_hist, color="blue")
        plt.title("Relativer Abstand ‖Δθ‖ / ‖θ‖")
        plt.xlabel("Iteration"); plt.ylabel("relative step"); plt.show()

        plt.figure()
        plt.plot(loss_before_hist, label="before", color="blue")
        plt.plot(loss_after_hist, label="after", color="magenta")
        plt.title("Complete Loss vor/nach Update")
        plt.xlabel("Iteration"); plt.ylabel("loss"); plt.legend(); plt.show()

        plt.figure()
        plt.plot(loss_delta_hist, color="blue")
        plt.title("Loss-Differenz (nach − vor)")
        plt.xlabel("Iteration"); plt.ylabel("Δloss"); plt.show()

        plt.figure()
        plt.plot(grad_l2_hist, color="blue")
        plt.title("Gradienten-L2-Norm"); plt.xlabel("Iteration"); plt.ylabel("‖g‖₂"); plt.show()

        plt.figure()
        plt.plot(grad_linf_hist, color="blue")
        plt.title("Gradienten-L∞-Norm"); plt.xlabel("Iteration"); plt.ylabel("‖g‖∞"); plt.show()

        plt.figure()
        plt.plot(θ_norm_hist, color="blue")
        plt.title("Parameternorm ‖θ‖"); plt.xlabel("Iteration"); plt.ylabel("‖θ‖"); plt.show()

        plt.figure()
        plt.plot(lr_hist)
        plt.title("Lernrate"); plt.xlabel("Iteration"); plt.ylabel("lr"); plt.show()

        print(f"Final Loss(θ0): {loss_before_hist[0]}")
        print(f"Final Loss(θ*): {loss_after_hist[-1]}")
        print(f"Absolute Final Error Reduction: {loss_after_hist[-1] - loss_before_hist[0]}")
        print(f"Relative Final Error Reduction: {((loss_after_hist[-1] - loss_before_hist[0])/loss_before_hist[0])*100:.2f}%")

    return θ, θ_history, grad_history

"""
Calculates the residual sum of squares for a given forecast dataset 
and the real value dataset.
"""
def rss(Y_hat, Y):
    return ((Y_hat - Y) ** 2).sum()

"""
Calculates the mean squared error for a given forecast dataset
and the real value dataset.

Specific loss function varies for the given dataset size. 
If dataset is the batch -> the loss is the Batch Loss
If dataset is whole training dataset -> empirical population risk
"""
def mse(Y_hat, Y):
    diff = Y_hat - Y
    return torch.mean(diff * diff)