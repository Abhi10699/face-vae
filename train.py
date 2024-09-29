import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import os

from model import VAE
from data_loader import dataloader

DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
EPOCHS = 200

vae_model = VAE(latent_dim=1024).to(device=DEVICE)

with torch.no_grad(): 
  sample_image = torch.randn((1,3, 64, 64)).to(DEVICE)
  model_out = vae_model(sample_image)
  print(model_out[0].shape)
  
   
  
optimizer = optim.Adam(vae_model.parameters())
# optimizer = torch.optim.adam.Adam(vae_model.parameters())

print("Training Started..")


kl_weight_min = 0.3  # Minimum KL weight
kl_weight_max = 1.0   # Maximum KL weight

def kl_weight_schedule(epoch):
  return kl_weight_min + (epoch / EPOCHS) * (kl_weight_max - kl_weight_min)

# start training
for epoch in range(EPOCHS):
  recon_loss_epoch = 0.0
  kl_loss_epoch = 0.0
  
  for idx, samples in enumerate(dataloader):
    optimizer.zero_grad()
    
    sample_img = samples.to(DEVICE)
    img_out, mu, log_var = vae_model(sample_img)
    
    # recon loss
    
    recon_loss = F.binary_cross_entropy(img_out, sample_img, reduction='sum')
  
    # kl_div loss
    # kl_weight = kl_weight_schedule(epoch)
    # kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
 
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    beta_kl_loss = kl_weight_schedule(epoch) * (2 * kl_divergence).mean()

    
    
    # total_loss = recon_loss + (kl_weight * kl_loss)
    
    total_loss = beta_kl_loss + recon_loss
    total_loss.backward()
    optimizer.step()


    recon_loss_epoch += recon_loss.item()
    kl_loss_epoch = beta_kl_loss.item()
  
    if idx % 500 == 0:
      print(f"\n\n[ITR]Recon Loss: {recon_loss.item()}, KL Loss: {beta_kl_loss.item()},")
      
  
  recon_loss_epoch /= len(dataloader)
  kl_loss_epoch /= len(dataloader)
  
  print(f"Recon Loss: {recon_loss_epoch}, KL Loss: {kl_loss_epoch},")
  
  with torch.no_grad():
    sample_image = samples[0].unsqueeze(0).to(DEVICE)  # Select the first image from the batch
    # random_sample = torch.randn((10, 128)).to(DEVICE)
    model_out = vae_model(sample_image)
    
    unnormalized_image = model_out[0]
    save_path = os.path.join("./training_images/", f"generated_epoch_{epoch+1}.png")
    vutils.save_image(unnormalized_image.detach().cpu(), save_path, normalize=False)
    print(f"Generated image saved at: {save_path}")



  # save model
  torch.save(vae_model.state_dict(), "./weights/model_weights_512_bottleneck.pth")