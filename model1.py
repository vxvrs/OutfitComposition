import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import transforms

from data_polyvore import PolyvoreDataset, Resize, ToTensor


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.lstm = nn.LSTM(42, 42)
        self.fc1 = nn.Linear(128 * 42 * 42, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 3)

        self.count = 0
        self.points = dict()

    def forward(self, x):
        img_cnn = list()
        for i, img in enumerate(x["image"]):
            x = img.type(torch.FloatTensor)
            x = self.pool(f.relu(self.conv1(x)))
            x = self.pool(f.relu(self.conv2(x)))
            # print(x.shape)
            img_cnn.append(x)

        # print(torch.cat(img_cnn).shape)
        x = torch.cat(img_cnn)

        # x, h = self.lstm(x)
        # print(x.shape)
        x = torch.flatten(x, 0)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)

        self.points[self.count] = x.detach()
        self.count += 1

        return x


def main():
    dataset = PolyvoreDataset("polyvore-dataset/train_no_dup.json",
                              transform=transforms.Compose([Resize(400), ToTensor()]))

    trainset = PolyvoreDataset("polyvore-dataset/test_no_dup.json",
                               transform=transforms.Compose([Resize(400), ToTensor()]))

    load = False
    model = Model()

    print(len(dataset))

    if not load:
        for i in range(1000):
            model(dataset[i])
        torch.save(model, "models/full1.pt")
    else:
        model = torch.load("models/full1.pt")
        model.eval()

    with torch.no_grad():
        print(model.points)

        sample = trainset[0]
        image = sample["image"][0]

        nd_images = np.zeros((8, 3, 400, 400), dtype=int)
        nd_images[0] = image

        out = model({"image": torch.from_numpy(nd_images)})
        plt.imshow(np.transpose(image, (1, 2, 0)))
        plt.show()

        dist = list()
        for idx, emb in model.points.items():
            dist.append((idx, torch.cdist(emb.unsqueeze(0), out.unsqueeze(0)).squeeze(0).numpy()[0]))

        sorted_dist = sorted(dist, key=lambda x: x[1])
        print(sorted_dist)
        closest = dataset[sorted_dist[1][0]]
        furthest = dataset[sorted_dist[-1][0]]

        for item in [closest, furthest]:
            for img in item["image"]:
                plt.imshow(np.transpose(img, (1, 2, 0)))
                plt.show()


if __name__ == "__main__":
    main()
