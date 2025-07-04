
torch.cuda.empty_cache()
losses_g=[]
losses_d=[]
real_scores=[]
fake_scores=[]

opt_d=torch.optim.Adam(params=discriminator.parameters(),lr=lr,betas=(0.5,0.999))
opt_g=torch.optim.Adam(params=generator.parameters(),lr=lr,betas=(0.5,0.999))
start_epoch=0
checkpoint_g=torch.load(r"/content/generator0_animeface.pth")
generator.load_state_dict(checkpoint_g["model_state_dict"])
opt_g.load_state_dict(checkpoint_g["optimizer_state_dict"])
start_epoch=checkpoint_g["epoch"]
loss_g=checkpoint_g["loss_g"]
checkpoint_d=torch.load(r"/content/discriminator0_animeface.pth")
discriminator.load_state_dict(checkpoint_d["model_state_dict"])
opt_d.load_state_dict(checkpoint_g["optimizer_state_dict"])
loss_d=checkpoint_d["loss_d"]
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

