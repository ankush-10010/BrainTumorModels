from torch import nn
class BrainTumorModelV2(nn.Module):
    def __init__(self,input_shape_flattened,input_channel,output_shape,hidden_units):
        super().__init__()
        self.conv_block_1=nn.Sequential(
            nn.Conv2d(in_channels=input_channel,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.Linear_layer_stack=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape_flattened,
                      out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape)

        )
    def forward(self,x):
        x=self.conv_block_1(x)
        # print(x.shape) torch.Size([16, 36, 36])
        x=self.conv_block_2(x)
        # print(x.shape) torch.Size([16, 18, 18])
        x=self.Linear_layer_stack(x)
        # print(x.shape) #torch.Size([16, 324])
        x=nn.Softmax(dim=1)(x)
        return x
# from torchvision import datasets,transforms
# from torch.utils.data import DataLoader

# directory_train_dataset=r"C:\CODE\Python\PytorchCourse\brain_tumore_datasets\Training"
# directory_test_dataset=r"C:\CODE\Python\PytorchCourse\brain_tumore_datasets\Testing"

# transform=transforms.Compose([
#     transforms.Resize((72,72)),
#     transforms.ToTensor()
# ])  

# train_dataset=datasets.ImageFolder(directory_train_dataset,
#                                    transform=transform)
# test_dataset=datasets.ImageFolder(directory_test_dataset,
#                                   transform=transform)

# BATCH_SIZE=32

# train_loader=DataLoader(dataset=train_dataset,
#                         batch_size=BATCH_SIZE,
#                         shuffle=True)

# test_loader=DataLoader(dataset=test_dataset,
#                        batch_size=BATCH_SIZE,
#                        shuffle=True)

# X_train,label=train_dataset[0]
# print(X_train.shape,X_train.unsqueeze(dim=0).shape)
# model_2=BrainTumorModelV2(input_shape_flattened=42,input_channel=3,
#                           output_shape=len(train_dataset.classes),
#                           hidden_units=16)

# model_2.forward(X_train)