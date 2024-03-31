import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from functools import partial

# VAE Configuration
input_size, hidden_size, latent_dim = 784, 400, 200

# Linear block
def linear_block(in_dim, out_dim, activation=nn.SiLU, normalization=nn.BatchNorm1d, use_bias=True):
    """
    Constructs a linear block with optional normalization and activation.
    """
    layers = nn.Sequential(nn.Linear(in_dim, out_dim, bias=use_bias))
    if activation: layers.append(activation())
    if normalization: layers.append(normalization(out_dim))
    return layers

# Initialize weights
def initialize_weights(module, leakiness=0.):
    """
    Initializes weights using Kaiming normal method with an optional leakiness parameter.
    """
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        init.kaiming_normal_(module.weight, a=leakiness)

weight_init = partial(initialize_weights, leakiness=0.2)  # Using a partial function for specific leakiness

# Variational Autoencoder
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(linear_block(input_size, hidden_size), linear_block(hidden_size, hidden_size))
        self.mean, self.log_var = linear_block(hidden_size, latent_dim, activation=None), linear_block(hidden_size, latent_dim, activation=None)
        self.decoder = nn.Sequential(linear_block(latent_dim, hidden_size), linear_block(hidden_size, hidden_size), linear_block(hidden_size, input_size, activation=None))
        self.apply(weight_init)  # Apply weight initialization
        
    def forward(self, x):
        encoded = self.encoder(x)
        mu, log_variance = self.mean(encoded), self.log_var(encoded)
        z = mu + (0.5 * log_variance).exp() * torch.randn_like(log_variance)  # Reparameterization trick
        return self.decoder(z), mu, log_variance
    
# KL Divergence Loss
def kl_divergence_loss(input, target):
    """
    Calculates the KL divergence loss component of VAE.
    """
    reconstructed, mu, log_variance = input
    return -0.5 * torch.sum(1 + log_variance - mu.pow(2) - log_variance.exp()) / mu.size(0)

# Binary Cross Entropy Loss
def reconstruction_loss(input, target):
    """
    Computes the binary cross entropy loss between the reconstructed and original images.
    """
    return F.binary_cross_entropy_with_logits(input[0], target, reduction='mean')

# Total VAE Loss
def total_vae_loss(input, target):
    """
    Computes the total VAE loss as the sum of KL divergence loss and reconstruction loss.
    """
    return kl_divergence_loss(input, target) + reconstruction_loss(input, target)
