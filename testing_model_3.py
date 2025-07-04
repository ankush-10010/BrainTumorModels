from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
from torch import nn
from helper_functions import accuracy_fn
directory_train_dataset=r"C:\CODE\Python\PytorchCourse\brain_tumore_datasets\Training"
directory_test_dataset=r"C:\CODE\Python\PytorchCourse\brain_tumore_datasets\Testing"

transform=transforms.Compose([
    # transforms.RandomRotation(30),
    # transforms.RandomGrayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.Resize((72,72)),
    transforms.ToTensor(),
])   
train_dataset=datasets.ImageFolder(directory_train_dataset,
                                   transform=transform)
test_dataset=datasets.ImageFolder(directory_test_dataset,
                                  transform=transform)

BATCH_SIZE=32

train_loader=DataLoader(dataset=train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True)

test_loader=DataLoader(dataset=test_dataset,
                       batch_size=BATCH_SIZE,
                       shuffle=True)
from model_3 import BrainTumorModelV3
X_train,labels=train_dataset[0]
model_3=BrainTumorModelV3(input_shape_flattened=1*5184,
                          input_channels=3,
                          output_shape=len(train_dataset.classes),hidden_units=16)
torch.manual_seed(42)
epochs=10
loss_fn=nn.CrossEntropyLoss()   
optimizer=torch.optim.SGD(params=model_3.parameters(),lr=0.1)  
start_epoch=0
checkpoint=torch.load(r"C:\CODE\Python\BrainTumorModels\trained_models_data\model_3_checkpoint.pth")
model_3.load_state_dict(checkpoint["model_state_dict"]) 
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch=checkpoint["epoch"]
print(f"Starting epoch:{start_epoch}")
loss=checkpoint["loss"]
best_acc=0
for epoch in tqdm(range(epochs)):
    train_loss=0
    for batch,(X_train,y_train) in enumerate(train_loader):
        model_3.train()
        y_pred=model_3(X_train)
        loss=loss_fn(y_pred,y_train)
        train_loss+=loss.item()
        optimizer.zero_grad()               
        loss.backward()
        optimizer.step()
        if batch%100==0:
            print(f"Looked at {batch*len(X_train)}/{len(train_loader.dataset)} samples")
    train_loss/=len(train_loader)
    test_loss=0
    test_acc=0
    model_3.eval()
    with torch.inference_mode():
        for X_test,y_test in test_loader:
            test_pred=model_3(X_test)
            test_loss+=loss_fn(test_pred,y_test).item()
            test_acc+=accuracy_fn(y_true=y_test,y_pred=test_pred.argmax(dim=1))
        test_loss/=len(test_loader)
        test_acc/=len(test_loader)
    print(f"\nEpoch:{epoch} | TrainLoss:{train_loss} | TestLoss:{test_loss} | TestAcc:{test_acc}")
    if test_acc>best_acc:
        best_acc=test_acc
        torch.save({
        "epoch":start_epoch+epoch,
        "model_state_dict":model_3.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "loss":loss.item(),
        },r"C:\CODE\Python\BrainTumorModels\trained_models_data\model_3_checkpoint.pth")
        print("Model saved successfully")
torch.save({
        "epoch":start_epoch+epoch,
        "model_state_dict":model_3.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "loss":loss.item(),
        },r"C:\CODE\Python\BrainTumorModels\trained_models_data\model_3_checkpoint.pth")
print("Model saved successfully")
