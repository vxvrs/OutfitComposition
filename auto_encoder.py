import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms

from data_polyvore import PolyvoreDataset, Resize, ToTensor, Normalize


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

        # self.fc = nn.Linear(1000, 625)
        # self.drop = nn.Dropout(0.2)
        #
        # self.flatten = nn.Flatten()
        # self.unflatten = nn.Unflatten(1, (1, 25, 25))

    def encoder(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = f.relu(self.conv3(x))
        x = self.pool(x)

        # x = self.flatten(x)

        return x

    def decoder(self, x):
        # x = self.unflatten(x)

        x = f.relu(self.t_conv1(x))
        x = f.relu(self.t_conv2(x))
        x = torch.sigmoid(self.t_conv3(x))

        return x

    def forward(self, x):
        x = x.type(torch.FloatTensor)

        x = self.encoder(x)
        x = self.decoder(x)

        return x


def main():
    trainset = PolyvoreDataset("polyvore-dataset/train_no_dup.json",
                               transform=transforms.Compose([Resize(400), Normalize(), ToTensor()]))
    validset = PolyvoreDataset("polyvore-dataset/valid_no_dup.json",
                               transform=transforms.Compose([Resize(400), Normalize(), ToTensor()]))

    trainloader = DataLoader(trainset, batch_size=50, shuffle=True, num_workers=1)

    validloader = DataLoader(validset, batch_size=50, shuffle=False, num_workers=1)

    print("Data loaded!", len(trainset), len(validset))

    model = AutoEncoder()
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    n_epochs = 5
    min_valid_loss = np.inf
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        model.train()

        for batch in trainloader:
            images = [img for outfit in batch["image"] for img in outfit]
            images = torch.from_numpy(np.stack(images, axis=0))
            if torch.cuda.is_available():
                images = images.cuda()

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, images.type(torch.FloatTensor))
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        valid_loss = 0.0
        model.eval()
        for batch in validloader:
            images = [img for outfit in batch["image"] for img in outfit]
            images = torch.from_numpy(np.stack(images, axis=0))
            if torch.cuda.is_available():
                images = images.cuda()

            outputs = model(images)
            loss = criterion(outputs, images.type(torch.FloatTensor))
            valid_loss = loss.item()

        print(
            f"Epoch {epoch} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(validloader)}")
        if min_valid_loss > valid_loss:
            print(f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model")
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), 'models/saved_model.pt')

        # for i in range(5):
        #     images = [img for img in trainset[i]["image"]]
        #     images = torch.from_numpy(np.stack(images, axis=0))
        #
        #     optimizer.zero_grad()
        #     outputs = model(images)
        #
        #     with torch.no_grad():
        #         for j, out in enumerate(outputs):
        #             img = np.transpose(out.detach().numpy(), (1, 2, 0))
        #             plt.imshow(img)
        #             plt.savefig(f"plots/e{epoch}_i{i}_o{j}.png")
        #
        #     loss = criterion(outputs, images.type(torch.FloatTensor))
        #     loss.backward()
        #     optimizer.step()
        #     train_loss += loss.item() * images.size(0)
        #
        #     print("Epoch: {} \tTraining Loss: {:.6f} \tTime: {:.2f}".format(epoch, train_loss, time.time() - epoch_start))


if __name__ == "__main__":
    main()
