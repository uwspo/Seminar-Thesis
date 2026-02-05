from NeuralNetwork import NN, MLP, TinyCNN
from SGD import stochastic_gradient_descent, rss, mse
from αEstimator import hill_α_estimator_Bootstrap_mse

from typing import List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

model = MLP(sizes=(1,3,3,1), activation=nn.ReLU, add_softmax=False)

#Synthetical Data for Training of the neural network
n = 80000
sigma_x = 3.0
sigma = 5.0
sigma_y = 3.0
d_input = 2

x0  = torch.normal(0.0, sigma_x, size=(d_input, 1))   
A   = torch.normal(0.0, sigma,   size=(n, d_input))  
eps = torch.normal(0.0, sigma_y, size=(n, 1))        

Y = A @ x0 + eps         
X = A @ x0

#Exogenous Parameters
η1, η2, η3, η4, η5, η6  = 0.01, 0.3, 0.1, 0.001, 0.0003, 0.01
b = 100

#Optimize model with sgd
θ0 = model.get_θ()

training_dataset = list(zip(X,Y))

θK, θ_iterates, grad_iterates = stochastic_gradient_descent(model, θ0, training_dataset, mse, η6, b, plot=True)

# Start the SGD out of the attractor θK
θK2, θ_iterates, grad_iterates = stochastic_gradient_descent(model, θK, training_dataset, mse, η6, b, plot=True)


#Calculate the fluctuations around the attractor θK

"""
Utility function that takes two Tensors 
and returns the component wise subtraction.
"""
def subtract_weights(weightsk: List[torch.Tensor], weightsK: List[torch.Tensor]) -> List[torch.Tensor]:
    if len(weightsk) != len(weightsK):
        raise ValueError("Tensors have different lengths.")
    return [weightk - weightK for weightk, weightK in zip(weightsk, weightsK)]

with torch.no_grad():
    ξ_iterates = [subtract_weights(θ, θK2) for θ in θ_iterates]    

#Calculate the Eigenvalues of the Hessian Matrix in the Attractor θK via the Power 
#Iteration Method

"""
Utility function that flattens a given tensor. 
"""
def _flatten(tlist: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tlist])

"""
Utility function that normalizes a vector. 
"""
def _normalize(v):
    nrm = _flatten(v).norm()
    return [p / (nrm + 1e-12) for p in v]

"""
Converts the input X to the forecast of the model.
The model uses the configuration, that is given with θ.
"""
def forecast(model, X, θ):
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
Calculates the loss for given θ.
The type of loss function varies with the given X,Y.
If X,Y is the whole training dataset -> empirical population risk.
If X,Y is just a custom Batch of the training dataset -> batch loss.
"""
def loss_function(model, θ, X, Y, loss):
    Y_hat = forecast(model, X, θ)
    return float(loss(Y_hat, Y).detach())

"""
Calculates the Gradient loss for a given θ.
The type of loss function varies with the given X,Y.
If X,Y is the whole training dataset -> gradient of the empirical population risk.
If X,Y is just a custom Batch of the training dataset -> batch gradient.
"""
def loss_function_grad(model, θ, X, Y, loss):
    θ_diff = [p.clone().detach().requires_grad_(True) for p in θ]
    Y_hat = forecast(model, X, θ_diff)
    loss_val = loss(Y_hat, Y)
    loss_val.backward()
    grads = [p.grad for p in θ_diff]
    return grads

"""
Linear Transformation H@v for a vector v.
enables use of Matrix Product with the hessian 
without calculating the Hessian Matrix itself.
"""
def hessian_vector_product(model, θ: List[torch.Tensor], X, Y, v):
    θ_diff = [p.clone().detach().requires_grad_(True) for p in θ]
    Y_hat = forecast(model, X, θ_diff)
    loss_val = mse(Y_hat, Y)

    grads = torch.autograd.grad(
        outputs=loss_val,
        inputs=θ_diff,
        create_graph=True,   
        retain_graph=True,  
        allow_unused=False
    )

    dot = sum((g * vi).sum() for g, vi in zip(grads, v))

    hvp = torch.autograd.grad(
        outputs=dot,
        inputs=θ_diff,
        retain_graph=False,  
        allow_unused=False
    )

    return hvp

"""
"""
def rayleigh_quotient(v: torch.Tensor, hvp: List[torch.Tensor]):
    v_flat = _flatten(v) if isinstance(v, list) else v.reshape(-1)
    hvp_flat = _flatten(hvp) if isinstance(hvp, (list, tuple)) else hvp.reshape(-1)
    return (torch.dot(v_flat, hvp_flat) / torch.dot(v_flat, v_flat))

"""
Algorithm that approximates the biggest Eigenvalue of the Hessian Matrix 
of the Empirical Population Risk on a given θ.
"""
def power_iteration_method(model, θ: List[torch.Tensor], tolerance: float):
    v = [torch.randn_like(p) for p in θ]
    v = _normalize(v)

    λ_old = None

    X_full = torch.stack([x for x, y in training_dataset])
    Y_full = torch.stack([y for x, y in training_dataset])

    while True:
        hvp = hessian_vector_product(model, θ, X_full, Y_full, v)

        λ = rayleigh_quotient(v, hvp).item()

        if λ_old is not None:
            if abs(λ - λ_old) < tolerance:
                break

        v = _normalize(list(hvp))
        λ_old = λ
        
    return λ, v

burn = int(0.6 * len(ξ_iterates))
thin = 8
ξ_used = ξ_iterates[burn::thin]

λ_max, v_max = power_iteration_method(model, θK2, 1e-5)

η_crit = 2*b / ((1 + b + 1) * λ_max)  

print("eta_crit≈", η_crit, "   eta_used≈", η6)


with torch.no_grad():
    T_list = [
        sum((x * u).sum() for x, u in zip(ξk, v_max))
        for ξk in ξ_used
    ]
rng = np.random.default_rng(1224) 
α_hat = hill_α_estimator_Bootstrap_mse(T_list, rng=rng)



def scaled_loss_factory(scale: float):
    def scaled_mse(yhat, y):
        return (scale ** 2) * mse(yhat, y)
    return scaled_mse

def hessian_vector_product_loss(model, θ: List[torch.Tensor], X, Y, v, loss_fn):
    θ_diff = [p.clone().detach().requires_grad_(True) for p in θ]
    Y_hat = forecast(model, X, θ_diff)
    loss_val = loss_fn(Y_hat, Y)

    grads = torch.autograd.grad(
        outputs=loss_val,
        inputs=θ_diff,
        create_graph=True,
        retain_graph=True,
        allow_unused=False
    )
    dot = sum((g * vi).sum() for g, vi in zip(grads, v))
    hvp = torch.autograd.grad(
        outputs=dot,
        inputs=θ_diff,
        retain_graph=False,
        allow_unused=False
    )
    return hvp

def power_iteration_method_loss(model, θ: List[torch.Tensor], tolerance: float, loss_fn):
    v = [torch.randn_like(p) for p in θ]
    v = _normalize(v)

    λ_old = None
    X_full = torch.stack([x for x, y in training_dataset])
    Y_full = torch.stack([y for x, y in training_dataset])

    while True:
        hvp = hessian_vector_product_loss(model, θ, X_full, Y_full, v, loss_fn)
        λ = rayleigh_quotient(v, hvp).item()
        if λ_old is not None:
            if abs(λ - λ_old) < tolerance:
                break
        v = _normalize(list(hvp))
        λ_old = λ
    return λ, v


def run_once_and_estimate(model_ctor, theta_start, dataset, eta, batchsize, loss_fn_for_sgd, 
                          loss_fn_for_hessian, burn_frac=0.6, thin=8, tol=1e-5, rng_seed=1224):
    model_local = model_ctor()

    θK, θ_iterates, _ = stochastic_gradient_descent(model_local, theta_start, dataset, mse, eta, batchsize, plot=False)

    θK2, θ_iterates2, _ = stochastic_gradient_descent(model_local, θK, dataset, loss_fn_for_sgd, eta, batchsize, plot=False)

    with torch.no_grad():
        ξ_iterates = [subtract_weights(θ, θK2) for θ in θ_iterates2]

    burn = int(burn_frac * len(ξ_iterates))
    ξ_used = ξ_iterates[burn::thin]

    λ_max, v_max = power_iteration_method_loss(model_local, θK2, tol, loss_fn_for_hessian)

    with torch.no_grad():
        T_list = [sum((x * u).sum() for x, u in zip(ξk, v_max)) for ξk in ξ_used]

    rng = np.random.default_rng(rng_seed)
    α_ret = hill_α_estimator_Bootstrap_mse(T_list, rng=rng)  # 5-Tupel
    return float(λ_max), α_ret[0]



def make_model():
    return MLP(sizes=(1,3,3,1), activation=nn.ReLU, add_softmax=False)

θ0_fresh = make_model().get_θ()
eta_fixed = η6  
b_fixed = b

curvature_scales = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

lambda_list = []
alpha_list = []

for c in curvature_scales:
    loss_sgd = scaled_loss_factory(c)          
    loss_hess = scaled_loss_factory(c)         

    λ_c, α_c = run_once_and_estimate(
        model_ctor=make_model,
        theta_start=θ0_fresh,
        dataset=training_dataset,
        eta=eta_fixed,
        batchsize=b_fixed,
        loss_fn_for_sgd=loss_sgd,
        loss_fn_for_hessian=loss_hess,
        burn_frac=0.6,
        thin=8,
        tol=1e-5,
        rng_seed=1224
    )
    lambda_list.append(λ_c)
    alpha_list.append(α_c)
    print(f"scale={c:>4}:  lambda_max≈{λ_c:.4f}   alpha_hat≈{α_c:.3f}")

plt.figure()
plt.scatter(lambda_list, alpha_list)

if len(lambda_list) >= 2:
    coef = np.polyfit(lambda_list, alpha_list, 1)
    xgrid = np.linspace(min(lambda_list), max(lambda_list), 100)
    ygrid = np.polyval(coef, xgrid)
    plt.plot(xgrid, ygrid)

plt.xlabel(r'$\lambda_{\max}$ (Krümmung)')
plt.ylabel(r'$\hat{\alpha}$ (Tail-Index)')
plt.title('Trend: stärkere Krümmung  →  kleinerer Tail-Index $\;\\hat{\\alpha}$')
plt.tight_layout()
plt.show()

