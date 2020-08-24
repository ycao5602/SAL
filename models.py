import torch
from torch import nn
import torch.nn.functional as F
import torchvision

'''
==============================================================================
model_g, model_d, model_x, model_a, model_i2,
model_ga, model_da, model_gx, model_dx, 
==============================================================================
'''

# generator for the attribute branch


class model_g(nn.Module):
    def __init__(self, input_size, output_size):

        super(model_g, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.linear2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.linear4 = nn.Linear(256, output_size)

    def forward(self, feature):

        f = self.linear1(feature)
        f = self.bn1(f)
        f = F.relu(f)
        f = self.linear2(f)
        f = self.bn2(f)
        f = F.relu(f)
        f = self.linear3(f)
        f = self.bn3(f)
        f = F.relu(f)
        f = self.linear4(f)
        output = torch.tanh(f)

        return output


# discriminator for the attribute branch


class model_d(nn.Module):

    def __init__(self, input_size):
        super(model_d, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, feature):
        f = self.linear1(feature)
        f = self.bn1(f)
        f = F.leaky_relu(f, 0.2, inplace=True)
        f = self.linear2(f)
        f = self.bn2(f)
        f = F.leaky_relu(f, 0.2, inplace=True)
        f = self.linear3(f)
        output = torch.sigmoid(f)

        return output


# encoder for the image branch


class model_x(nn.Module):

    def __init__(self, input_size, output_size):
        super(model_x, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, output_size)

    def forward(self, feature):
        f = self.linear1(feature)
        f = self.bn1(f)
        f = F.relu(f)
        f = self.linear2(f)
        f = self.bn2(f)
        f = F.relu(f)
        f = self.linear3(f)
        output = torch.tanh(f)

        return output


# encoder for the attribute branch


class model_a(nn.Module):

    def __init__(self, input_size, output_size):
        super(model_a, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, output_size)

    def forward(self, feature):
        f = self.linear1(feature)
        f = self.bn1(f)
        f = F.relu(f)
        f = self.linear2(f)
        f = self.bn2(f)
        f = F.relu(f)
        f = self.linear3(f)
        output = torch.tanh(f)

        return output


# shared classifier


class model_i2(nn.Module):

    def __init__(self, input_size, output_size):
        super(model_i2, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)

    def forward(self, feature):
        output = self.linear1(feature)

        return output


# generator for the fake attribute features a'


class model_ga(nn.Module):
    def __init__(self, input_size, output_size):

        super(model_ga, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.linear4 = nn.Linear(256, output_size)

    def forward(self, feature):

        f = self.linear1(feature)
        f = self.bn1(f)
        f = F.relu(f)
        f = self.linear2(f)
        f = self.bn2(f)
        f = F.relu(f)
        f = self.linear3(f)
        f = self.bn3(f)
        f = F.relu(f)
        f = self.linear4(f)
        output = torch.tanh(f)

        return output


# generator for the fake image features x'


class model_gx(nn.Module):
    def __init__(self, input_size, output_size):

        super(model_gx, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.linear4 = nn.Linear(256, output_size)

    def forward(self, feature):

        f = self.linear1(feature)
        f = self.bn1(f)
        f = F.relu(f)
        f = self.linear2(f)
        f = self.bn2(f)
        f = F.relu(f)
        f = self.linear3(f)
        f = self.bn3(f)
        f = F.relu(f)
        f = self.linear4(f)
        output = torch.tanh(f)

        return output


# discriminator for a' and a


class model_da(nn.Module):

    def __init__(self, input_size):
        super(model_da, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 1)

    def forward(self, feature):
        f = self.linear1(feature)
        f = self.bn1(f)
        f = F.leaky_relu(f, 0.2, inplace=True)
        f = self.linear2(f)
        f = self.bn2(f)
        f = F.leaky_relu(f, 0.2, inplace=True)
        f = self.linear3(f)
        output = torch.sigmoid(f)

        return output


# discriminator for x' and x


class model_dx(nn.Module):

    def __init__(self, input_size):
        super(model_dx, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 1)

    def forward(self, feature):
        f = self.linear1(feature)
        f = self.bn1(f)
        f = F.leaky_relu(f, 0.2, inplace=True)
        f = self.linear2(f)
        f = self.bn2(f)
        f = F.leaky_relu(f, 0.2, inplace=True)
        f = self.linear3(f)
        output = torch.sigmoid(f)

        return output