# Variational Autoencoder
import torch
import torch.nn as nn
import torch.nn.functional as F

GLOBAL_KERNEL_SIZE = 4
GLOBAL_STRIDE_SIZE = 2
GLOBAL_PADDING_SIZE = 1

class Encoder(nn.Module):
  def __init__(self, latent_dim = 128):
    super().__init__()
    
    self.kernel_size = GLOBAL_KERNEL_SIZE
    self.stride = GLOBAL_STRIDE_SIZE
    self.latent_dim = latent_dim
    self.padding = GLOBAL_PADDING_SIZE
    
    
    self.encoder_seq = nn.Sequential(
      nn.Conv2d(3, 16, self.kernel_size, self.stride, self.padding),
      nn.LeakyReLU(0.2),
      
      nn.Conv2d(16, 32, self.kernel_size, self.stride, self.padding),
      nn.LeakyReLU(0.2),
     
      nn.Conv2d(32, 64, self.kernel_size, self.stride, self.padding),
      nn.LeakyReLU(0.2),
    
      nn.Conv2d(64, 128, self.kernel_size, self.stride, self.padding),
      nn.LeakyReLU(0.2),
  
      nn.Flatten() #
    )

    self.encoder_out_shape = 2048

    self.bottle_neck = nn.Linear(self.encoder_out_shape, self.latent_dim)
    self.mu = nn.Linear(self.latent_dim,self.latent_dim)
    self.log_var = nn.Linear(self.latent_dim,self.latent_dim)
  
  
  
  def forward(self, x):
    x  = self.encoder_seq(x)
    x = F.relu(self.bottle_neck(x))
    
    
    x_mu = self.mu(x)
    x_logvar = self.log_var(x)
    
    x_sample = self.sample(x_mu, x_logvar)
    return x, x_sample, x_mu, x_logvar

  def sample(self, mu: torch.Tensor, log_var: torch.Tensor):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)
  
class Decoder(nn.Module):
  def __init__(self, latent_dim=128):
    super().__init__()
    
    self.kernel_size = GLOBAL_KERNEL_SIZE
    self.stride = GLOBAL_STRIDE_SIZE
    self.padding = GLOBAL_PADDING_SIZE
    
    
    self.fc1 = nn.Linear(latent_dim, 2048)
    
    self.decoder_seq = nn.Sequential(  
  
      nn.ConvTranspose2d(128, 64, self.kernel_size, self.stride, self.padding),
      nn.LeakyReLU(0.2),
      
      nn.ConvTranspose2d(64, 32, self.kernel_size, self.stride, self.padding),
      nn.LeakyReLU(0.2),
      
      nn.ConvTranspose2d(32, 16, self.kernel_size, self.stride, self.padding),
      nn.LeakyReLU(0.2),
      
      nn.ConvTranspose2d(16, 3, self.kernel_size, self.stride, self.padding),
      nn.Sigmoid(),
    )
  
  def forward(self, x):
    x = self.fc1(x).view((-1, 128, 4, 4 ))
    return self.decoder_seq(x)
  
  def decode_from_encoder(self, x):
    x = x.view((-1, 128, 4, 4))
    return self.decoder_seq(x)
  
    
class VAE(nn.Module):
  def __init__(self, latent_dim=128):
    super().__init__()
    self.encoder = Encoder(latent_dim=latent_dim)
    self.decoder = Decoder(latent_dim=latent_dim)
    
    # latent vars

  def forward(self, x):
    x_encoded, x_sample, x_mu, x_logvar = self.encoder(x)
    # prepare for decoder
    x_out = self.decoder(x_sample)
    return x_out, x_mu, x_logvar
  
class Discriminator(nn.Module):
    def __init__(self):
      super(Discriminator, self).__init__()
      self.disc = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # More channels
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(128),
        
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Increase depth
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # Single output for real/fake classification
        nn.Flatten(),
        nn.Sigmoid()  # Sigmoid for binary classification
      )
    
    def forward(self, x):
      x = self.disc(x)
      return x
