from torch import nn
class BrainTumorModelV3(nn.Module):
    def __init__(self,input_shape_flattened,input_channels,output_shape,hidden_units):
        super().__init__()
        self.expansion=4
        self.ResNET=nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                             out_channels=hidden_units,
                             kernel_size=3,
                             stride=1,
                             padding=1),
            nn.BatchNorm2d(num_features=hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                              out_channels=hidden_units,
                              kernel_size=3,
                              stride=1,
                              padding=1),
            nn.BatchNorm2d(num_features=hidden_units),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,
                      kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=hidden_units),
            nn.ReLU(),
            
        )
        self.conv_block_1=nn.Sequential(
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
            nn.MaxPool2d(kernel_size=2)
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
        x=self.ResNET(x) 
        # print(x.shape) #torch.Size([1, 16, 72, 72])
        x=self.conv_block_1(x) 
        # print(x.shape) #torch.Size([1, 16, 36, 36])
        x=self.conv_block_2(x)
        # print(x.shape) #torch.Size([1, 16, 18, 18])
        x=self.Linear_layer_stack(x)
        # print(x.shape) #torch.Size([1, 5184]) (for flattened)
        return x

##Test_accuracy is around 92-94% for Hidden_units 16 and just 4 basic transforms
