import torch.nn as nn

class Penguin12Classifier(nn.Module):
    def __init__(self, in_features=1024, num_classes=1):
        super(Penguin12Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),  # concat -> 512*2
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)
