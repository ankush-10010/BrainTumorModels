import torch
import torch.optim.sgd
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch import nn
from helper_functions import accuracy_fn
from tqdm.auto import tqdm

directory_train_dataset=r"C:\CODE\Python\PytorchCourse\brain_tumore_datasets\Training"
directory_test_dataset=r"C:\CODE\Python\PytorchCourse\brain_tumore_datasets\Testing"

transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0,translate=(0.1,0.1)),
    transforms.Resize((72,72)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
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
from model_2 import BrainTumorModelV2
X_train_0,label_0=train_dataset[0]
print(X_train_0.shape)
print(len(train_dataset.classes))

model_2=BrainTumorModelV2(input_shape_flattened=16*324,input_channel=3,hidden_units=16,output_shape=len(train_dataset.classes))

torch.manual_seed(42)
epochs=10
loss_fn=nn.CrossEntropyLoss()   
optimizer=torch.optim.SGD(params=model_2.parameters(),lr=0.0003)  
start_epoch=0
checkpoint=torch.load(r"C:\CODE\Python\BrainTumorModels\trained_models_data\model_2_checkpoint1.pth")
model_2.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch=checkpoint["epoch"]
print(f"Starting epoch:{start_epoch}")
loss=checkpoint["loss"]
best_acc=0
for epoch in tqdm(range(epochs)):
    train_loss=0
    for batch,(X_train,y_train) in enumerate(train_loader):
        model_2.train()
        y_pred=model_2(X_train)
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
    model_2.eval()
    with torch.inference_mode():
        for X_test,y_test in test_loader:
            test_pred=model_2(X_test)
            test_loss+=loss_fn(test_pred,y_test).item()
            test_acc+=accuracy_fn(y_true=y_test,y_pred=test_pred.argmax(dim=1))
        test_loss/=len(test_loader)
        test_acc/=len(test_loader)
    print(f"\nEpoch:{epoch} | TrainLoss:{train_loss} | TestLoss:{test_loss} | TestAcc:{test_acc}")
    if test_acc>best_acc:
        best_acc=test_acc
        torch.save({
        "epoch":start_epoch+epoch,
        "model_state_dict":model_2.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "loss":loss.item(),
        },r"C:\CODE\Python\BrainTumorModels\trained_models_data\model_2_checkpoint1.pth")
        print("Model saved successfully")

# torch.save({
#     "epoch":start_epoch+epochs,
#     "model_state_dict":model_2.state_dict(),
#     "optimizer_state_dict":optimizer.state_dict(),
#     "loss":loss.item(),
# }, r"C:\CODE\Python\BrainTumorModels\trained_models_data\model_2_checkpoint.pth")

# print("Model saved successfully")
