import torch
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import os

from diffusers import VQModel
from data_loader import dataloader

DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
EPOCHS = 40

model = VQModel(
  in_channels=3,
  out_channels=3,
  layers_per_block=4,
  sample_size=64
).to(DEVICE)

model.load_state_dict(torch.load("./model_weights_512.pth"))
  
optimizer = optim.Adam(model.parameters())
latent_code = torch.randn((1, 3))


def train():
  for epoch in range(EPOCHS):
    recon_loss_epoch = 0.0
    kl_loss_epoch = 0.0
    
    for idx, samples in enumerate(dataloader):
      optimizer.zero_grad()
      
      sample_img = samples.to(DEVICE)
      img_out = model(sample_img)
      
      # recon loss
      
      recon_loss = F.mse_loss(img_out.sample, sample_img)
      total_loss = recon_loss + img_out.commit_loss

      total_loss.backward()
      optimizer.step()


      recon_loss_epoch += recon_loss.item()
    
      if idx % 500 == 0:
        print(f"[ITR] Recon Loss: {recon_loss.item()}")
        

    recon_loss_epoch /= len(dataloader)
    kl_loss_epoch /= len(dataloader)
    
    print(f"Recon Loss: {recon_loss_epoch}, KL Loss: {kl_loss_epoch},")
    
    # save model
    torch.save(model.state_dict(), "./model_weights_512.pth")
      
def predict():
  with torch.no_grad():
    for epoch in range(10):
      latent_height = 64  # Adjust based on your model's configuration
      latent_width = 64   # Adjust based on your model's configuration
      batch_size = 1

      num_embeddings = model.quantize.n_e

      indices = torch.randint(0, num_embeddings, (batch_size, latent_height, latent_width)).to(DEVICE)

      one_hot = F.one_hot(indices, num_embeddings).float()

      latent = one_hot @ model.quantize.embedding.weight  # Shape: (batch_size, latent_height, latent_width, embedding_dim)
      latent = latent.permute(0, 3, 1, 2)  # Shape: (batch_size, embedding_dim, latent_height, latent_width)

      decoded = model.decode(latent)
      
      save_path = os.path.join("./training_images/", f"generated_epoch_{epoch+1}.png")
      vutils.save_image(decoded.sample.detach().cpu(), save_path, normalize=False)
      print(f"Generated image saved at: {save_path}")

  
  
predict()