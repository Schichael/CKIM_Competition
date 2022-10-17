import torch
from federatedscope.register import register_model

class MyNet(torch.nn.Module):
    def __init__(self,
                 in_channels=1,
                 h=32,
                 w=32,
                 hidden=2048,
                 class_num=10,
                 use_bn=True):
        super(MyNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = torch.nn.Linear((h // 2 // 2) * (w // 2 // 2) * 64, hidden)
        self.fc2 = torch.nn.Linear(hidden, class_num)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(self.relu(x))
        x = self.conv2(x)
        x = self.maxpool(self.relu(x))
        x = torch.nn.Flatten()(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_my_net(model_config, local_data):
    # You can also build models without local_data
    data = next(iter(local_data['train']))
    model = MyNet(in_channels=data[0].shape[1],
                  h=data[0].shape[2],
                  w=data[0].shape[3],
                  hidden=model_config.hidden,
                  class_num=model_config.out_channels)
    return model


def call_my_net(model_config, local_data):
    if model_config.type == "mynet2":
        model = load_my_net(model_config, local_data)
        return model


# register_model("mynet", call_my_net)