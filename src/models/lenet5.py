from torch import nn

class leNet5(nn.Module):

    def __init__(self, learning_rate=1e-3, momentum=0.9):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.feature = nn.Sequential(
            # first convolutional layer - input 3*32*32 -> output 64*28*28 
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*14*14
            
            # second convolutional layer - input 64*14*14 -> output 64*10*10 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*5*5
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64*5*5, out_features=384),
            nn.ReLU(),
            nn.Linear(in_features=384, out_features=192),
            nn.ReLU(),
            nn.Linear(in_features=192, out_features=100),
        )
        
    def forward(self, x):
        return self.classifier(self.feature(x))
