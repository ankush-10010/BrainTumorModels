from torchvision import datasets,transforms
from torch.utils.data import DataLoader
directory_train_dataset=r"C:\CODE\Python\PytorchCourse\brain_tumore_datasets\Training"
directory_test_dataset=r"C:\CODE\Python\PytorchCourse\brain_tumore_datasets\Testing"

transform=transforms.Compose([
    transforms.Resize((72,72)),
    transforms.ToTensor()
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