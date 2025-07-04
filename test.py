from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from torch import nn
from helper_functions import accuracy_fn
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset

directory_test_dataset=r"C:\CODE\Python\BrainTumorModels\brain_tumore_datasets\Testing"
transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.Resize((72,72)),
    transforms.ToTensor(),
])  
test_dataset=datasets.ImageFolder(directory_test_dataset,
                                  transform=transform,)
BATCH_SIZE=32

test_loader=DataLoader(dataset=test_dataset,
                       batch_size=BATCH_SIZE,
                       shuffle=True)

from model_3 import BrainTumorModelV3
model_3=BrainTumorModelV3(input_shape_flattened=1*5184,
                          input_channels=3,
                          output_shape=len(test_dataset.classes),hidden_units=16)
torch.manual_seed(42)
loss_fn=nn.CrossEntropyLoss()
# optimizer=torch.optim.SGC(params=model_2.parameters(),lr=0.01)
checkpoint=torch.load(r"C:\CODE\Python\BrainTumorModels\trained_models_data\model_3_checkpoint.pth")
model_3.load_state_dict(checkpoint["model_state_dict"])
start_epoch=checkpoint["epoch"]
epochs=100
test_loss=0
test_acc=0
for epoch in tqdm(range(epochs)):
    model_3.eval()
    with torch.inference_mode():
        for X_test,y_test in test_loader:
            test_pred=model_3(X_test)
            test_loss+=loss_fn(test_pred,y_test)
            test_acc+=accuracy_fn(y_true=y_test,y_pred=test_pred.argmax(dim=1))
        test_loss/=len(test_loader)
        test_acc/=len(test_loader)
    print(f"\nEpoch:{epoch} | TestLoss:{test_loss} | TestAcc:{test_acc}")
    