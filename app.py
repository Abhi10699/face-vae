import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np

from io import BytesIO
from flask import Flask, render_template, request, send_file
from model import VAE
from PIL import Image
from data_loader import preprocess



DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"

vae_model = VAE(latent_dim=1024).to(device=DEVICE)
vae_model.load_state_dict(torch.load("./weights/model_weights_512_bottleneck.pth", weights_only=True))
vae_model.eval()

app = Flask(__name__, template_folder="./templates")
app.config['TEMPLATES_AUTO_RELOAD'] = True


def linear_interpolate(latent_start, latent_end, steps=10):
  interpolated_latents = []
  for alpha in torch.linspace(0, 1, steps):
      latent = (1 - alpha) * latent_start + alpha * latent_end
      interpolated_latents.append(latent)
  return interpolated_latents

@app.get("/")
def index():
  # sample_image = torch.randn((20,512)).to(DEVICE)
  # model_out = vae_model.decoder(sample_image)[0].detach().cpu().numpy() * 255.0
  # image = np.transpose(model_out, (1, 2, 0)).astype(np.uint8)
  # image = Image.fromarray(image)
  
  # img_io = BytesIO()
  # image.save(img_io, 'PNG')
  # img_io.seek(0)

    # Send the image to the frontend
  # return send_file(img_io, mimetype='image/png')
  return render_template("index.html")


@app.post('/interpolate')
def interpolate():

  face_1 = request.files['face1']
  face_2 = request.files['face2']
  
  face_1_img = preprocess(Image.open(face_1.stream)).unsqueeze(0).to(DEVICE)
  face_2_img = preprocess(Image.open(face_2.stream)).unsqueeze(0).to(DEVICE)
  
  # get encoded representations
  
  face_1_encoded = vae_model.encoder(face_1_img)[1]
  face_2_encoded = vae_model.encoder(face_2_img)[1]

  # interplolate
  interplolated_latents = linear_interpolate(face_1_encoded, face_2_encoded, steps=150)
  decoded_samples = []
  
  for space in interplolated_latents:
  
    decoded_sample = vae_model.decoder(space)[0].detach().cpu().numpy() * 255.0
    decoded_sample_np = np.transpose(decoded_sample, (1, 2 ,0)).astype(np.uint8)
    decoded_sample_img = Image.fromarray(decoded_sample_np).resize((256, 256), resample=4)
    
    decoded_samples.append(decoded_sample_img)
  
  # decode all interpolations
  
  img_io = BytesIO()  
  decoded_samples[0].save(img_io,"GIF",
               save_all=True, append_images=decoded_samples[1:], optimize=True, duration=40, loop=0)
  img_io.seek(0)

  return send_file(img_io, mimetype='image/gif')