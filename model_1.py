from torch import nn
class BrainTumorModelV1(nn.Module):
    def __init__(self,input_shapes,hidden_units,output_shapes):
        super().__init__()
        self.conv_block_1=nn.Sequential(
            nn.Conv2d(in_channels=input_shapes,
            out_channels=hidden_units,
            kernel_size=3,
            stride=1,
            padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
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
            out_channels=output_shapes,
            kernel_size=3,
            stride=1,
            padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1296,
                    out_features=output_shapes)
        )
    def forward(self,x):
        x=self.conv_block_1(x)
        # print(x.shape)
        x=self.conv_block_2(x)
        # print(x.shape)
        x=self.classifier(x)
        # print(x.shape)
        return x
    
"""
AROUND 10 EPOCHS ACCURACY OF ONLY 30%

"""