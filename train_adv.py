import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from model import VAE, Discriminator
from data_loader import dataloader
from torchvision.utils import save_image


# setup training device

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_INTERVAL = 200


vae_base = VAE(latent_dim=512).to(DEVICE)
vae_base.load_state_dict(torch.load("./face_model_vae_gan_4x4_512_bottleneck.pth", weights_only=True))

# load discriminator

discriminator_model = Discriminator().to(DEVICE)
with torch.no_grad():
  sample_img = torch.randn((1, 3, 64, 64)).to(DEVICE)
  d_out = discriminator_model(sample_img)
  print(d_out.shape)

# setup optimizers

vae_optim = optim.Adam(vae_base.parameters(), lr=1e-4)
discriminator_optim = optim.Adam(discriminator_model.parameters(), lr=1e-5)

# train

EPOCHS = 15000

kl_weight_max = 1.0
kl_weight_schedule = lambda epoch: min(kl_weight_max, epoch / EPOCHS)

for epoch in range(EPOCHS):
  
  running_recon_loss = 0.0
  running_kl_loss = 0.0
  running_discrim_loss = 0.0
  running_total_loss = 0.0
  running_discrim_real_loss = 0.0
  running_discrim_fake_loss = 0.0
      
  for idx, data in enumerate(dataloader):
    # clear grads
    vae_optim.zero_grad()
    discriminator_model.zero_grad()
    
    # load data on training device
    
    data = data.to(DEVICE)
    
    # train discriminator on real images
    
    discrim_out = discriminator_model(data)
    discrim_real_label = torch.ones((data.shape[0], 1)).to(DEVICE)
    discrim_loss_real = F.binary_cross_entropy(discrim_out, discrim_real_label, reduction='sum')
    discrim_loss_real.backward()
    
    # get vae outputs
    
    img_out, mu, log_var = vae_base(data)
    recon_loss = F.binary_cross_entropy(img_out, data, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
    # train discriminator on fake images
    
    discrim_fake_out = discriminator_model(img_out.detach())
    discrim_fake_label = torch.zeros((data.shape[0], 1)).to(DEVICE)
    discrim_loss_fake = F.binary_cross_entropy(discrim_fake_out, discrim_fake_label, reduction='sum')
    discrim_loss_fake.backward()

    discriminator_optim.step()
    
    # train vae on adv loss
    
    discrim_out_vae = discriminator_model(img_out)
    discrim_real_label = torch.ones((data.shape[0], 1)).to(DEVICE)
    discrim_vae_loss = F.binary_cross_entropy(discrim_out_vae, discrim_real_label, reduction='sum')
    
    
    kl_weight = kl_weight_schedule(epoch)
    total_loss = recon_loss + (kl_weight * kl_loss) + (0.1 * discrim_vae_loss)
    
    total_loss.backward()
    vae_optim.step()    

    
    running_recon_loss += recon_loss.item()
    running_kl_loss += kl_loss.item()
    running_discrim_loss += discrim_vae_loss.item()
    running_total_loss += total_loss.item()
    
    running_discrim_real_loss = discrim_loss_real.item()
    running_discrim_fake_loss = discrim_loss_fake.item()
    
    if (idx % LOG_INTERVAL) == 0:
      print(
        f'Epoch [{epoch+1}/{EPOCHS}], Iteration [{idx}], '
        f'Recon Loss: {running_recon_loss / LOG_INTERVAL:.4f}, '
        f'KL Loss: {running_kl_loss / LOG_INTERVAL:.4f}, '
        f'Discrim VAE Loss: {running_discrim_loss / LOG_INTERVAL:.4f}, ',
        f'Discrim Loss (real): {running_discrim_real_loss / LOG_INTERVAL:.4f}, ',
        f'Discrim Loss (fake): {running_discrim_fake_loss / LOG_INTERVAL:.4f}, ',
        f'Total Loss: {running_total_loss / LOG_INTERVAL:.4f}'
      )
      
      running_recon_loss = 0.0
      running_kl_loss = 0.0
      running_discrim_loss = 0.0
      running_total_loss = 0.0

  with torch.no_grad():
    sample_img_out, _, _ = vae_base(data[:20])  # Take the first 8 samples in the batch for visualization
    save_image(sample_img_out, f'./training_images/sample_epoch_{epoch+1}.png')    
    
    # Save model after each epoch
  torch.save(vae_base.state_dict(), f'./face_model_vae_gan_4x4_512_bottleneck.pth')
  print(f"Model saved for epoch {epoch+1}. Image saved as './training_images/sample_epoch_{epoch+1}.png'.")
