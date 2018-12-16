import torch.nn as nn

class Linear(nn.Module):

    def __init__(self, feature_dim, num_classes=10):
        super(Linear, self).__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

        

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
