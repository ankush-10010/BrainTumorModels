

import kagglehub
# Download latest version
path = kagglehub.dataset_download("aadilmalik94/animecharacterfaces")

print("Path to dataset files:", path)

import os
DATA_DIR="/root/.cache/kagglehub/datasets/aadilmalik94/animecharacterfaces/versions/1"
print(os.listdir(DATA_DIR))

print(os.listdir(DATA_DIR+"/animeface-character-dataset/data"))

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

image_size=64
batch_size=128
stats=(0.5,0.5,0.5),(0.5,0.5,0.5)

train_ds=ImageFolder(DATA_DIR,transform=T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(*stats)
]
))
train_dl=DataLoader(train_ds,
                    batch_size,
                    shuffle=True,
                    num_workers=3,
                    pin_memory=True)

# Commented out IPython magic to ensure Python compatibility.
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
# %matplotlib inline

def denorm(img_tensors):
  return img_tensors * stats[1][0] + stats[0][0]

def show_images(images,nmax=64):
  fig,ax=plt.subplots(figsize=(8,8))
  ax.set_xticks([]);ax.set_yticks([])
  ax.imshow(make_grid(denorm(images.detach()[:nmax]),nrow=8).permute(1,2,0))

def show_batch(dl):
  for images,_ in dl:
    show_images(images,64)
    break

show_batch(train_dl)

def get_default_device():
  if torch.cuda.is_available():
    return torch.device('cuda')
  else:
    return torch.device('cpu')
def to_device(data,device):
  if isinstance(data,(list,tuple)):
    return [to_device(x,device) for x in data]
  return data.to(device,non_blocking=True)

class DeviceDataLoader():
  def __init__(self,dl,device):
    self.dl=dl
    self.device=device
  def __iter__(self):
    for b in self.dl:
      yield to_device(b,self.device)
  def __len__(self):
    return len(self.dl)

device=get_default_device()
device

train_dl=DeviceDataLoader(train_dl,device)

import torch.nn as nn

discriminator=nn.Sequential(
    #in:3*64*64
    nn.Conv2d(in_channels=3,
              out_channels=64,
              kernel_size=4,
              stride=2,
              padding=1,
              bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(negative_slope=0.2,
                 inplace=True),
    #out:64*32*32

    nn.Conv2d(in_channels=64,
              out_channels=128,
              kernel_size=4,
              stride=2,
              padding=1,
              bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(negative_slope=0.2,
                 inplace=True),
    #out:128*16*16

    nn.Conv2d(in_channels=128,
              out_channels=256,
              kernel_size=4,
              stride=2,
              padding=1,
              bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(negative_slope=0.2,
                 inplace=True),
    #out:256*8*8

    nn.Conv2d(in_channels=256,
              out_channels=512,
              kernel_size=4,
              stride=2,
              padding=1,
              bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(negative_slope=0.2,
                 inplace=True),
    #out:512*4*4

    nn.Conv2d(in_channels=512,
              out_channels=1,
              kernel_size=4,
              stride=1,
              padding=0,
              bias=False),
    #out:1*1*1
    nn.Flatten(),
    nn.Sigmoid()
)

discriminator=to_device(discriminator,device)

latent_size=128

generator=nn.Sequential(
    #in:latent_size*1*1
    nn.ConvTranspose2d(in_channels=latent_size,
                       out_channels=512,
                       kernel_size=4,
                       stride=1,
                       padding=0,
                       bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True),
    #out:512*4*4
    nn.ConvTranspose2d(in_channels=512,
                       out_channels=256,
                       kernel_size=4,
                       stride=2,
                       padding=1,
                       bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True),
    #out:256*8*8
    nn.ConvTranspose2d(in_channels=256,
                       out_channels=128,
                       kernel_size=4,
                       stride=2,
                       padding=1,
                       bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    #out:128*16*16
    nn.ConvTranspose2d(in_channels=128,
                       out_channels=64,
                       kernel_size=4,
                       stride=2,
                       padding=1,
                       bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    #out:64*32*32
    nn.ConvTranspose2d(in_channels=64,
                       out_channels=3,
                       kernel_size=4,
                       stride=2,
                       padding=1,
                       bias=False),
    nn.Tanh()
    #out:3*64*64

)

xb=torch.randn(batch_size,latent_size,1,1)
fake_images=generator(xb)
print(fake_images.shape)
show_images(fake_images)

generator=to_device(generator,device)

def train_discriminator(real_images,opt_d):
  opt_d.zero_grad()
  real_preds=discriminator(real_images)
  real_targets=torch.ones(real_images.size(0),1,device=device)
  real_loss=nn.functional.binary_cross_entropy(input=real_preds,
                                               target=real_targets,
                                               weight=None)
  real_score=torch.mean(real_preds).item()

  latent=torch.randn(batch_size,latent_size,1,1,device=device)
  fake_images=generator(latent)

  fake_targets=torch.zeros(fake_images.size(0),1, device=device)
  fake_preds=discriminator(fake_images)
  fake_loss=nn.functional.binary_cross_entropy(input=fake_preds,
                                               target=fake_targets,
                                               weight=None)
  fake_score=torch.mean(fake_preds).item()

  loss=real_loss+fake_loss
  loss.backward()
  opt_d.step()
  return loss.item(),real_score,fake_score

def train_generator(opt_g):
  opt_g.zero_grad()

  latent=torch.randn(batch_size,latent_size,1,1,device=device)
  fake_images=generator(latent)

  preds=discriminator(fake_images)
  targets=torch.ones(size=(fake_images.size(0),1),device=device)
  loss=nn.functional.binary_cross_entropy(input=preds,
                                          target=targets,
                                          weight=None)
  loss.backward()
  opt_g.step()
  return loss.item()

from torchvision.utils import save_image

sample_dir="generated"
os.makedirs(sample_dir,exist_ok=True)

def save_samples(index,latent_tensors,show=True):
  fake_images=generator(latent_tensors)
  fake_fname="generated-images-{0:0=4d}.png".format(index)
  save_image(denorm(fake_images),os.path.join(sample_dir,fake_fname),nrow=8)
  print("Saving",fake_fname)
  if show:
    fig,ax=plt.subplots(figsize=(8,8))
    ax.set_xticks([]);ax.set_yticks([])
    ax.imshow(make_grid(fake_images.cpu().detach(),nrow=8).permute(1,2,0))

fixed_latent=torch.randn(64,latent_size,1,1,device=device)

save_samples(index=0,latent_tensors=fixed_latent)

from tqdm.notebook import tqdm
import torch.nn.functional as f
import torch.nn as nn

# def fit(epochs,lr,start_index=1):
#   torch.cuda.empty_cache()
#   losses_g=[]
#   losses_d=[]
#   real_scores=[]
#   fake_scores=[]

#   opt_d=torch.optim.Adam(params=discriminator.parameters(),lr=lr,betas=(0.5,0.999))
#   opt_g=torch.optim.Adam(params=generator.parameters(),lr=lr,betas=(0.5,0.999))
#   start_epoch=0
#   checkpoint_g=torch.load(r"/content/generator0_animeface.pth")
#   generator.load_state_dict(checkpoint_g["model_state_dict"])
#   opt_g.load_state_dict(checkpoint_g["optimizer_state_dict"])
#   start_epoch=checkpoint_g["epoch"]
#   loss_g=checkpoint_g["loss_g"]
#   checkpoint_d=torch.load(r"/content/discriminator0_animeface.pth")
#   discriminator.load_state_dict(checkpoint_d["model_state_dict"])
#   opt_d.load_state_dict(checkpoint_g["optimizer_state_dict"])
#   loss_d=checkpoint_d["loss_d"]
#   for epoch in range(epochs):
#     for real_images,_ in tqdm(train_dl):
#       loss_d,real_score,fake_score=train_discriminator(real_images=real_images,opt_d=opt_d)
#       loss_g=train_generator(opt_g)

#     losses_g.append(loss_g)
#     losses_d.append(loss_d)
#     real_scores.append(real_score)
#     fake_scores.append(fake_score)

#     print("Epoch [{}/{}] , loss_g:{:.4f} , loss_d:{:.4f} , real_scores:{:.4f} ,fake_score:{:.4f}".format(epoch+1,epochs,loss_g,loss_d,real_score,fake_score))

#     save_samples(epoch+start_index,latent_tensors=fixed_latent,show=False)

#   return losses_g,losses_d,real_scores,fake_scores

epochs=2
lr=0.0002

torch.cuda.empty_cache()
losses_g=[]
losses_d=[]
real_scores=[]
fake_scores=[]
start_index=1
opt_d=torch.optim.Adam(params=discriminator.parameters(),lr=lr,betas=(0.5,0.999))
opt_g=torch.optim.Adam(params=generator.parameters(),lr=lr,betas=(0.5,0.999))
start_epoch=0
checkpoint_g=torch.load(r"/content/generator0_animeface.pth")
generator.load_state_dict(checkpoint_g["model_state_dict"])
opt_g.load_state_dict(checkpoint_g["optimizer_state_dict"])
start_epoch=checkpoint_g["epoch"]
loss_g=checkpoint_g["loss"]
checkpoint_d=torch.load(r"/content/discriminator0_animeface.pth")
discriminator.load_state_dict(checkpoint_d["model_state_dict"])
opt_d.load_state_dict(checkpoint_g["optimizer_state_dict"])
loss_d=checkpoint_d["loss"]
for epoch in range(epochs):
  for real_images,_ in tqdm(train_dl):
    loss_d,real_score,fake_score=train_discriminator(real_images=real_images,opt_d=opt_d)
    loss_g=train_generator(opt_g)

  losses_g.append(loss_g)
  losses_d.append(loss_d)
  real_scores.append(real_score)
  fake_scores.append(fake_score)

  print("Epoch [{}/{}] , loss_g:{:.4f} , loss_d:{:.4f} , real_scores:{:.4f} ,fake_score:{:.4f}".format(epoch+1,epochs,loss_g,loss_d,real_score,fake_score))

  save_samples(epoch+start_index,latent_tensors=fixed_latent,show=False)

torch.save({
        "epoch":start_epoch+epoch,
        "model_state_dict":generator.state_dict(),
        "optimizer_state_dict":opt_g.state_dict(),
        "loss":loss_g,
        },r"/content/generator0_animeface.pth")
torch.save({
        "epoch":start_epoch+epoch,
        "model_state_dict":discriminator.state_dict(),
        "optimizer_state_dict":opt_d.state_dict(),
        "loss":loss_d,
        },r"/content/discriminator0_animeface.pth")

print(losses_g,losses_d,real_scores,fake_scores)