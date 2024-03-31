import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from datasets import load_dataset
from fastprogress import progress_bar
from einops import rearrange

# Load dataset
data_label, label_name = 'image', 'label'
dataset_name = "fashion_mnist"
batch_size = 512
dataset = load_dataset(dataset_name)

# Timestep embedding for temporal information in UNet
def time_embed(steps, dim, period_max=10000):
    """
    Creates sinusoidal embeddings based on the input time steps.
    """
    rates = -math.log(period_max) * torch.linspace(0, 1, dim // 2, device=steps.device)
    embedding = steps[:, None].float() * rates.exp()[None, :]
    embedding = torch.cat([embedding.sin(), embedding.cos()], dim=-1)
    return F.pad(embedding, (0, 1, 0, 0)) if dim % 2 == 1 else embedding

# Preprocessing convolutional block
def conv_block(in_dim, out_dim, kernel=3, stride=1, activation=nn.SiLU, normalization=None, use_bias=True):
    """
    A basic convolutional block with optional normalization and activation.
    """
    layers = nn.Sequential()
    if normalization: layers.append(normalization(in_dim))
    if activation: layers.append(activation())
    layers.append(nn.Conv2d(in_dim, out_dim, stride=stride, kernel_size=kernel, padding=kernel // 2, bias=use_bias))
    return layers

def upsample_block(filters): 
    return nn.Sequential(nn.Upsample(scale_factor=2.), nn.Conv2d(filters, filters, 3, padding=1))

def linear_block(in_dim, out_dim, activation=nn.SiLU, normalization=None, use_bias=True):
    """
    A basic linear block with optional normalization and activation.
    """
    layers = nn.Sequential()
    if normalization: layers.append(normalization(in_dim))
    if activation: layers.append(activation())
    layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
    return layers

class SelfAttn(nn.Module):
    def __init__(self, dim, attn_dim, transpose=True):
        super().__init__()
        self.heads = dim // attn_dim
        self.scale = math.sqrt(dim / self.heads)
        self.norm = nn.LayerNorm(dim)
        self.qkv_gen = nn.Linear(dim, dim * 3)
        self.project = nn.Linear(dim, dim)
        self.trans = transpose
    
    def forward(self, input):
        n, _, s = input.shape
        if self.trans: input = input.transpose(1, 2)
        input = self.norm(input)
        qkv = self.qkv_gen(input)
        qkv = rearrange(qkv, 'n s (h d) -> (n h) s d', h=self.heads)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        scores = (q @ k.transpose(1, 2)) / self.scale
        attn = scores.softmax(dim=-1) @ v
        attn = rearrange(attn, '(n h) s d -> n s (h d)', h=self.heads)
        output = self.project(attn)
        if self.trans: output = output.transpose(1, 2)
        return output

class EmbResBlk(nn.Module):
    def __init__(self, emb_dim, in_dim, out_dim=None, kernel=3, activation=nn.SiLU, norm=nn.BatchNorm2d, attn_dim=0):
        """
        An embodied residual block with optional self-attention.
        """
        super().__init__()
        if out_dim is None: out_dim = in_dim
        self.emb_proj = nn.Linear(emb_dim, out_dim * 2)
        self.conv1 = conv_block(in_dim, out_dim, kernel, activation=activation, normalization=norm)
        self.conv2 = conv_block(out_dim, out_dim, kernel, activation=activation, normalization=norm)
        self.id_conv = nn.Identity() if in_dim == out_dim else nn.Conv2d(in_dim, out_dim, 1)
        self.attn = None
        if attn_dim: self.attn = SelfAttn(out_dim, attn_dim, transpose=False)

    def forward(self, input, t_emb):
        """
        Forward pass with embedded timestep and optional self-attention.
        """
        x = input
        x = self.conv1(x)
        emb = self.emb_proj(F.silu(t_emb))[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)
        x = x * (1 + scale) + shift
        x = self.conv2(x)
        x = x + self.id_conv(input)
        if self.attn: x = x + self.attn(x)
        return x

# Modify forward methods to intercept and save intermediate results for residual connections
def save_intermediate(module, save_to):
    original_forward = module.forward

    def forward_hook(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        save_to.append(result)
        return result

    module.forward = forward_hook
    return module

class DownBlk(nn.Module):
    def __init__(self, emb_dim, in_dim, out_dim, add_down=True, layers=1, attn_dim=0):
        """
        Downscaling block with optional self-attention and convolution.
        """
        super().__init__()
        self.res_blks = nn.ModuleList([save_intermediate(EmbResBlk(emb_dim, in_dim if i == 0 else out_dim, out_dim, attn_dim=attn_dim), self)
                                       for i in range(layers)])
        self.downsample = save_intermediate(nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1), self) if add_down else nn.Identity()

    def forward(self, input, t_emb):
        self.saved = []
        for res_blk in self.res_blks:
            input = res_blk(input, t_emb)
        input = self.downsample(input)
        return input

class UpBlk(nn.Module):
    def __init__(self, emb_dim, in_dim, prev_out_dim, out_dim, add_up=True, layers=2, attn_dim=0):
        """
        Upscaling block with optional self-attention for UNet architecture.
        """
        super().__init__()
        self.res_blks = nn.ModuleList([EmbResBlk(emb_dim, (prev_out_dim if i == 0 else out_dim) + (in_dim if (i == layers - 1) else out_dim), out_dim, attn_dim=attn_dim)
                                       for i in range(layers)])
        self.upsample = upsample_block(out_dim) if add_up else nn.Identity()

    def forward(self, input, t_emb, up_inputs):
        """
        Forward pass through upscaling block with concatenation of skip connections.
        """
        for res_blk in self.res_blks:
            input = res_blk(torch.cat([input, up_inputs.pop()], dim=1), t_emb)
        return self.upsample(input)

class EmbUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, filters=(224, 448, 672, 896), layers=1, attn_chans=8, attn_start=1):
        super().__init__()
        self.conv_start = nn.Conv2d(in_ch, filters[0], kernel_size=3, padding=1)
        self.emb_dim = filt = filters[0]
        n_emb = filt * 4
        self.emb_mlp = nn.Sequential(linear_block(self.emb_dim, n_emb, normalization=nn.BatchNorm1d),
                                     linear_block(n_emb, n_emb))
        self.down_layers = nn.ModuleList()
        num_filters = len(filters)
        for i in range(num_filters):
            in_filt = filt
            filt = filters[i]
            self.down_layers.append(DownBlk(n_emb, in_filt, filt, add_down=i != num_filters - 1, layers=layers,
                                            attn_dim=0 if i < attn_start else attn_chans))
        self.mid = EmbResBlk(n_emb, filters[-1])

        rev_filters = list(reversed(filters))
        filt = rev_filters[0]
        self.up_layers = nn.ModuleList()
        for i in range(num_filters):
            prev_filt = filt
            filt = rev_filters[i]
            in_filt = rev_filters[min(i + 1, len(filters) - 1)]
            self.up_layers.append(UpBlk(n_emb, in_filt, prev_filt, filt, add_up=i != num_filters - 1, layers=layers + 1,
                                        attn_dim=0 if i >= num_filters - attn_start else attn_chans))
        self.conv_end = conv_block(filters[0], out_ch, activation=nn.SiLU, normalization=nn.BatchNorm2d, use_bias=False)

    def forward(self, inputs):
        x, t = inputs
        t_emb = time_embed(t, self.emb_dim)
        emb = self.emb_mlp(t_emb)
        x = self.conv_start(x)
        saved = [x]
        for block in self.down_layers:
            x = block(x, emb)
        saved += [p for block in self.down_layers for p in block.saved]
        x = self.mid(x, emb)
        for block in self.up_layers:
            x = block(x, emb, saved)
        return self.conv_end(x)

# Cosine scheduler for noise levels
def cosine_scheduler(t):
    return (t * math.pi / 2).cos() ** 2

def inverse_cosine_scheduler(x):
    return x.sqrt().acos() * 2 / math.pi

# Add noise to the input images
def add_noise(input_images):
    device = input_images.device
    num_images = len(input_images)
    t_random = torch.rand(num_images, device=device).clamp(0, 0.999)
    noise = torch.randn(input_images.shape, device=device)
    cos_t = cosine_scheduler(t_random).reshape(-1, 1, 1, 1).to(device)
    noisy_images = cos_t.sqrt() * input_images + (1 - cos_t).sqrt() * noise
    return (noisy_images, t_random.to(device)), noise

# Custom collate function for DDPM DataLoader
def ddpm_collate(batch):
    return add_noise(batch[data_label])

# DataLoader for DDPM
def create_ddpm_loader(dataset):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=ddpm_collate, num_workers=4)

def ddim_step(x_t, epsilon, alpha_bar_t, alpha_bar_t1, beta_bar_t, beta_bar_t1, eta, sigma, apply_clamp=True):
    """
    Performs a DDIM step for iterative refinement of the image.
    """
    sigma_adjusted = ((beta_bar_t1 / beta_bar_t).sqrt() * (1 - alpha_bar_t / alpha_bar_t1).sqrt()) * eta
    x_0_estimated = ((x_t - (1 - alpha_bar_t).sqrt() * epsilon) / alpha_bar_t.sqrt())
    if apply_clamp: x_0_estimated = x_0_estimated.clamp(-1, 1)
    if beta_bar_t1 <= sigma ** 2 + 0.01: sigma = 0.
    x_next_t = alpha_bar_t1.sqrt() * x_0_estimated + (beta_bar_t1 - sigma ** 2).sqrt() * epsilon
    x_next_t += sigma * torch.randn(x_next_t.shape).to(x_next_t)
    return x_0_estimated, x_next_t

# Sample generation with DDIM
@torch.no_grad()
def generate_samples(func, model, image_size, steps, eta=1., apply_clamp=True):
    """
    Generates samples using DDIM over a specified number of steps.
    """
    model.eval()
    time_steps = torch.linspace(1 - 1 / steps, 0, steps)
    x_t = torch.randn(image_size).cuda()
    predictions = []
    for i, t in enumerate(progress_bar(time_steps)):
        t = t[None].cuda()
        alpha_bar_t = cosine_scheduler(t)
        epsilon = model((x_t, t))
        alpha_bar_t1 = cosine_scheduler(t - 1 / steps) if t >= 1 / steps else torch.tensor(1)
        x_0_hat, x_t = func(x_t, epsilon, alpha_bar_t, alpha_bar_t1, 1 - alpha_bar_t, 1 - alpha_bar_t1, eta, 1 - ((i + 1) / 100), apply_clamp=apply_clamp)
        predictions.append(x_0_hat.float().cpu())
    return predictions

class CondUNet(nn.Module):
    def __init__(self, num_classes, in_ch=3, out_ch=3, filters=(224, 448, 672, 896), layers=1):
        """
        Conditional U-Net model that allows for class-conditional image generation.
        """
        super().__init__()
        self.conv_start = nn.Conv2d(in_ch, filters[0], kernel_size=3, padding=1)
        self.emb_dim = filt = filters[0]
        n_emb = filt * 4
        self.class_emb = nn.Embedding(num_classes, n_emb)
        self.emb_mlp = nn.Sequential(linear_block(self.emb_dim, n_emb, normalization=nn.BatchNorm1d),
                                     linear_block(n_emb, n_emb))
        self.down_layers = nn.ModuleList()
        for i in range(len(filters)):
            in_filt = filt
            filt = filters[i]
            self.down_layers.append(DownBlk(n_emb, in_filt, filt, add_down=i != len(filters) - 1, layers=layers))
        self.mid = EmbResBlk(n_emb, filters[-1])

        rev_filters = list(reversed(filters))
        filt = rev_filters[0]
        self.up_layers = nn.ModuleList()
        for i in range(len(filters)):
            prev_filt = filt
            filt = rev_filters[i]
            in_filt = rev_filters[min(i + 1, len(filters) - 1)]
            self.up_layers.append(UpBlk(n_emb, in_filt, prev_filt, filt, add_up=i != len(filters) - 1, layers=layers + 1))
        self.conv_end = conv_block(filters[0], out_ch, activation=nn.SiLU, normalization=nn.BatchNorm2d, use_bias=False)

    def forward(self, inputs):
        x, t, c = inputs
        t_emb = time_embed(t, self.emb_dim)
        c_emb = self.class_emb(c)
        emb = self.emb_mlp(t_emb) + c_emb
        x = self.conv_start(x)
        saved = [x]
        for block in self.down_layers:
            x = block(x, emb)
        saved += [p for block in self.down_layers for p in block.saved]
        x = self.mid(x, emb)
        for block in self.up_layers:
            x = block(x, emb, saved)
        return self.conv_end(x)

# Conditional sample generation with DDIM
@torch.no_grad()
def conditional_generate_samples(class_label, func, model, image_size, steps, eta=1.):
    time_steps = torch.linspace(1 - 1 / steps, 0, steps)
    x_t = torch.randn(image_size).cuda()
    c = x_t.new_full((image_size[0],), class_label, dtype=torch.int32)
    predictions = []
    for i, t in enumerate(progress_bar(time_steps)):
        t = t[None].cuda()
        alpha_bar_t = cosine_scheduler(t)
        epsilon = model((x_t, t, c))
        alpha_bar_t1 = cosine_scheduler(t - 1 / steps) if t >= 1 / steps else torch.tensor(1)
        x_0_hat, x_t = func(x_t, epsilon, alpha_bar_t, alpha_bar_t1, 1 - alpha_bar_t, 1 - alpha_bar_t1, eta, 1 - ((i + 1) / 100))
        predictions.append(x_0_hat.float().cpu())
    return predictions
