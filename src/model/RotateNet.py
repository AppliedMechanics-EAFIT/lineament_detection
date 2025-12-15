import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# Adpatation of A convolutional neural network for semi-automated lineament detection 
# and vectorisation of remote sensing data using probabilistic clustering: A method and a challenge
# Amin Aghaee, Pejman Shamsipour, Shawn Hood, Rasmus Haugaard
# https://doi.org/10.1016/j.cageo.2021.104724

class RotateNet(nn.Module):
    def __init__(self, w, in_channels=8, hidden_units=300):
        super(RotateNet, self).__init__()
        self.conv = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        # MaxPool para reducir de 49 -> 43
        self.pool = nn.MaxPool2d(kernel_size=7, stride=1)
        
        # Tamaño después de pooling: 43 x 43 x 8
        flattened_size = 8 * 43 * 43
        
        self.fc1 = nn.Linear(flattened_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 1)
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W = 49
    model = RotateNet(w=W, in_channels=8).to(device)
    summary(model, input_size=(8, 49, 49))