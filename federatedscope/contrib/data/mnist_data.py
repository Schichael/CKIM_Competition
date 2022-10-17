from federatedscope.register import register_data
import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def load_my_data(config):


    # Build data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
    data_train = MNIST(root='data', train=True, transform=transform, download=True)
    data_test = MNIST(root='data', train=False, transform=transform, download=True)

    # Split data into dict
    data_dict = dict()
    train_per_client = len(data_train) // config.federate.client_num
    test_per_client = len(data_test) // config.federate.client_num
    train_per_client = train_per_client // 60
    print(f"train per client: {train_per_client}")
    for client_idx in range(1, config.federate.client_num + 1):
        dataloader_dict = {
            'train':
                DataLoader([
                    data_train[i]
                    for i in range((client_idx - 1) *
                                   train_per_client, client_idx * train_per_client)
                ],
                    config.data.batch_size,
                    shuffle=config.data.shuffle),
            'test':
                DataLoader([
                    data_test[i]
                    for i in range((client_idx - 1) * test_per_client, client_idx *
                                   test_per_client)
                ],
                    config.data.batch_size,
                    shuffle=False)
        }
        data_dict[client_idx] = dataloader_dict

    return data_dict, config



def call_my_data(config):
    if config.data.type == "mydata":
        data, modified_config = load_my_data(config)
        return data, modified_config


# register_data("mydata", call_my_data)

