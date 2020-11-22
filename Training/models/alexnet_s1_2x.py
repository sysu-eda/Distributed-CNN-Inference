import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


# you need to download the models to ~/.torch/models
# model_urls = {
#     'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
# }
models_dir = os.path.expanduser('~/.torch/models')
model_name = 'alexnet-owt-4df8aa71.pth'

class AlexNet_S(nn.Module):
    def __init__(self, num_classes=1000, num_group=4):
        super(AlexNet_S, self).__init__()
        
        self.num_group = num_group

        self.Conv1 = nn.Sequential(
            # input shape is 224 x 224 x 3
            nn.Conv2d(3, 128, kernel_size=11, stride=4, padding=2), # shape is 55 x 55 x 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2) # shape is 27 x 27 x 64
        )
        
        self.GConv2 = nn.Sequential(
            nn.Conv2d(128, 384, kernel_size=5, padding=2, groups=num_group), # shape is 27 x 27 x 192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # shape is 13 x 13 x 192
        )
        
        self.GConv3 = nn.Sequential(
            nn.Conv2d(384, 768, kernel_size=3, padding=1, groups=num_group), # shape is 13 x 13 x 384
            nn.ReLU(inplace=True)
        )

        self.GConv4 = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1, groups=num_group), # shape is 13 x 13 x 256
            nn.ReLU(inplace=True)
        )
        
        self.GConv5 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, groups=num_group), # shape is 13 x 13 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2) # shape is 6 x 6 x 256
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        #x = self.features(x)
        x=self.Conv1(x)
        x=self.GConv2(x)
        x=self.GConv3(x)
        x=self.channel_shuffle(x)
        x=self.GConv4(x)
        x=self.GConv5(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
    
    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.num_group == 0
        group_channels = num_channels // self.num_group
        
        x = x.reshape(batchsize, group_channels, self.num_group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)
        return x

def alexnet_s1_2x(pretrained=False, **kwargs):
    """
    AlexNet model architecture 

    Args:
        pretrained (bool): if True, returns a model pre-trained on ImageNet
    """
    model = AlexNet_S(**kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name)))
    return model