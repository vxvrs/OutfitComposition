import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms

from data import PolyvoreDataset, Resize, ToTensor, Normalize


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.t_conv1 = nn.ConvTranspose2d(1, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(8, 3, 2, stride=2)
        # TODO: Add dense layers to improve embedding.
        flatten_size = 1 * 50 * 50
        # self.fc = nn.Linear(flatten_size, flatten_size)
        # self.t_fc = nn.Linear(flatten_size, flatten_size)
        self.drop = nn.Dropout(0.2)

        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (1, 50, 50))

    def encoder(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = f.relu(self.conv3(x))
        x = self.pool(x)

        x = self.flatten(x)

        # x = f.relu(self.fc(x))
        # x = self.drop(x)

        return x

    def decoder(self, x):
        # x = f.relu(self.t_fc(x))

        x = self.unflatten(x)

        x = f.relu(self.t_conv1(x))
        x = f.relu(self.t_conv2(x))
        x = torch.sigmoid(self.t_conv3(x))

        return x

    def forward(self, x):
        x = x.type(torch.FloatTensor)

        x = self.encoder(x)
        print("Shape after encoding: ", x.shape)
        x = self.decoder(x)
        print("Shape after decoding: ", x.shape)

        return x


def train_epoch():
    pass


def main():
    main_start = time.time()

    trainset = PolyvoreDataset("polyvore-dataset/train_no_dup.json",
                               transform=transforms.Compose([Resize(400), Normalize(), ToTensor()]))

    trainloader = DataLoader(trainset, batch_size=50, shuffle=True, num_workers=1)

    print("Data loaded!", len(trainset))

    model = AutoEncoder()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # images = list()
    # for i in range(len(trainset)):
    #     image = trainset[i]["image"]
    #     for img in image:
    #         images.append(img)
    #
    # data = [torch.unsqueeze(img, 0) for img in images]
    # data = torch.cat(data)
    # print(data.shape)

    n_epochs = 200
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        train_loss = 0.0

        # for batch in trainloader:
        #     images = [img for outfit in batch["image"] for img in outfit]

        for i in range(5):
            images = [img for img in trainset[i]["image"]]
            images = torch.from_numpy(np.stack(images, axis=0))

            optimizer.zero_grad()
            outputs = model(images)

            with torch.no_grad():
                for j, out in enumerate(outputs):
                    img = np.transpose(out.detach().numpy(), (1, 2, 0))
                    plt.imshow(img)
                    plt.savefig(f"plots/e{epoch}_i{i}_o{j}.png")

            loss = criterion(outputs, images.type(torch.FloatTensor))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        print("Epoch: {} \tTraining Loss: {:.6f} \tTime: {:.2f}".format(epoch, train_loss, time.time() - epoch_start))

    print("Total time: {:.2f}".format(time.time() - main_start))


if __name__ == "__main__":
    main()
