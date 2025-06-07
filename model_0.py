from torch import nn
class BrainTumorModelV0(nn.Module):
    def __init__(self,input_shapes,hidden_units,output_shapes):
        super().__init__()
        self.layer_stack=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shapes,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=output_shapes)
        )
    def forward(self,x):
        return self.layer_stack(x)
    
"""
AROUND 30 EPOCHS IT GIVES ACCURACY OF 73%

"""
