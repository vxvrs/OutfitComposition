import time

import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import transforms

from data import PolyvoreDataset, Resize, ToTensor, Normalize


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)
        # TODO: Add dense layers for embedding.
        # self.d1 = nn.Linear(4 * 100 * 100, )

    def encoder(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)

        return x

    def decoder(self, x):
        x = f.relu(self.t_conv1(x))
        x = torch.sigmoid(self.t_conv2(x))

        return x

    def forward(self, x):
        x = x.type(torch.FloatTensor)

        x = self.encoder(x)
        x = self.decoder(x)

        return x


def main():
    main_start = time.time()

    trainset = PolyvoreDataset("polyvore-dataset/train_no_dup.json",
                               transform=transforms.Compose([Resize(400), Normalize(), ToTensor()]))

    model = AutoEncoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    images = list()
    for i in range(len(trainset)):
        image = trainset[i]["image"]
        for img in image:
            images.append(img)

    data = [torch.unsqueeze(img, 0) for img in images]
    data = torch.cat(data)
    print(data.shape)

    n_epochs = 10
    for epoch in range(1, n_epochs + 1):
        start_time = time.time()
        train_loss = 0.0
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data.type(torch.FloatTensor))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

        print("Epoch: {} \tTraining Loss: {:.6f} \tTime: {:.2f}".format(epoch, train_loss, time.time() - start_time))

    print("Total time: {:.2f}".format(time.time() - main_start))


if __name__ == "__main__":
    main()