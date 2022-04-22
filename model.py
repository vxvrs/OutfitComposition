import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from data import PolyvoreDataset, Resize, ToTensor


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.lstm = nn.LSTM(42, 42)
        self.fc1 = nn.Linear(128 * 42 * 42, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 2)

    def forward(self, x):
        img_cnn = list()
        for img in x["image"]:
            x = img.type(torch.FloatTensor)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            # print(x.shape)
            img_cnn.append(x)

        # print(torch.cat(img_cnn).shape)
        x = torch.cat(img_cnn)

        x, h = self.lstm(x)
        # print(x.shape)
        x = torch.flatten(x, 0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    dataset = PolyvoreDataset("polyvore-dataset/train_no_dup.json",
                              transform=transforms.Compose([Resize(400), ToTensor()]))

    sample_img = dataset[0]["image"][0]
    sample_img2 = dataset[0]["image"][1]
    other_img = dataset[777]["image"][0]

    load = True
    model = Model()

    if not load:
        for i in range(200):
            print(i)
            model(dataset[i])
        torch.save(model, "model1.pt")
    else:
        model = torch.load("model1.pt")
        model.eval()

    # tensor([-0.0349, -0.0028])
    # tensor([-0.0333, -0.0077])

    with torch.no_grad():
        results = list()

        for sample in [sample_img, sample_img2, other_img]:
            nd_images = np.zeros((8, 3, 400, 400), dtype=int)
            nd_images[0] = sample
            images = torch.from_numpy(nd_images)

            results.append(model({"image": images}))

        # results = np.array(results)
        # print(torch.from_numpy(results))
        print(results)
        for r in results:
            print(torch.cdist(results[0], r))


if __name__ == "__main__":
    main()
