import torch
import torch.nn as nn
from torchvision import models
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class DMInet(nn.Module):
    def __init__(self, model_name = 'resnet34', cls_num = 4):
        super(DMInet, self).__init__()

        exec("self.model = models.{}(pretrained=True)".format(model_name))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = self.model_rivise(model_name)
        self.classfier= nn.Linear(in_features, cls_num)
        
    def model_rivise(self, model_name):
        if model_name in ["resnet18", "resnet34", "resnet50", "resnet101"]:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            in_features = self.model.fc.in_features
            # self.model = nn.Sequential(*list(self.model.children())[:-2])
        elif model_name == "densenet161":
            self.model.features.conv0 = nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=3, bias=False)
            in_features = self.model.classifier.in_features  # 2208
            # self.model = self.model.features
        return in_features

    def forward(self, data):
        x = self.model(data)
        x = self.pool(x).squeeze() # bs * in_features
        out = self.classfier(x)
        return out


# from torchsummary import summary
# if __name__=="__main__":

#     model = DMInet(model_name= 'resnet34')
#     model = model.cuda()
#     summary(model, input_size=(1,224,224), batch_size=20, device='cuda')