# Image Generation with Stable Diffusion

This repository contains an implementation focused on image generation using stable diffusion techniques. Our project leverages a combination of deep learning architectures including Variational Autoencoders (VAE), U-Net, along with noise schedulers and samplers such as Denoising Diffusion Probabilistic Models (DDPM) and Denoising Diffusion Implicit Models (DDIM) to generate high-quality images.

## Key Features

- **VAE**: Utilized for learning efficient data encodings.
- **U-Net**: Adapted for the generation phase, ensuring detailed and coherent image outputs.
- **Noise Schedulers**: Implements various noise scheduling techniques crucial for the diffusion process.
- **Samplers**: Includes DDPM and DDIM for sampling and generating final images from the learned distribution.

## Installation

Ensure you have Python 3.6 or later installed.

### tinytorchutil Package

This project makes use of `tinytorchutil` ([Git-Repo](https://github.com/Chainathan/tiny-torch-util)), a personal toy package containing a collection of utility functions found useful. Install it using pip:

```bash
pip install tinytorchutil
```

---
