import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PolyvoreDataset(Dataset):
    def __init__(self, json_filename, transform=None):
        self.data = self.load_data(json_filename)
        self.transform = transform
        self.word_idx, self.idx_word = self.__word_to_idx()
        self.__convert_names()
        self.__fix_sizes()

    def load_data(self, filename):
        with open(filename) as f:
            json_data = json.load(f)

        filtered_data = list()
        for data in json_data:
            outfit = dict([(key, list()) for key in ["name", "categoryid", "image_filename"]])

            for item in data["items"]:
                if item["name"] != "polyvore" and item["name"] != "":
                    outfit["name"].append(' '.join(item["name"].split(' ')[:10]))
                    outfit["categoryid"].append(item["categoryid"])
                    image_filename = f"polyvore-dataset/images/{data['set_id']}/{item['index']}.jpg"
                    outfit["image_filename"].append(image_filename)

            outfit["name"] = np.array(outfit["name"])
            outfit["categoryid"] = np.array(outfit["categoryid"])
            outfit["image_filename"] = np.array(outfit["image_filename"])
            filtered_data.append(outfit)

        return np.array(filtered_data)

    def __word_to_idx(self):
        name_count = dict()
        for outfit in self.data:
            for name in outfit["name"]:
                for word in name.split(' '):
                    if word not in name_count:
                        name_count[word] = 1
                    else:
                        name_count[word] += 1

        index = 1
        word_idx = dict()
        idx_word = dict()
        for word, count in name_count.items():
            if count > 30:
                word_idx[word] = index
                idx_word[index] = word
                index += 1

        return word_idx, idx_word

    def __convert_names(self):
        for outfit in self.data:
            new_names = [np.array([self.word_idx[word] for word in name.split() if word in self.word_idx]) for name in
                         outfit["name"]]
            outfit["name"] = new_names

    def __fix_sizes(self):
        for i, outfit in enumerate(self.data):
            names = outfit["name"]
            for j, name in enumerate(names):
                if len(name) < 10:  # Item descriptions have a maximum of 10 words.
                    for _ in range(10 - len(name)):
                        names[j] = np.append(names[j], 0)

            if len(names) < 8:  # Outfit has a maximum of 8 items
                for _ in range(8 - len(names)):
                    names.append(np.array([0] * 10))

            self.data[i]["name"] = np.array(names)

            categoryid = outfit["categoryid"]
            if len(categoryid) < 8:
                for _ in range(8 - len(categoryid)):
                    categoryid = np.append(categoryid, 0)

            self.data[i]["categoryid"] = categoryid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.data[idx]["name"]
        categoryid = self.data[idx]["categoryid"]
        image_filename = self.data[idx]["image_filename"]
        image = [cv2.imread(filename)[..., ::-1] for filename in image_filename]

        sample = {"name": name, "categoryid": categoryid, "image": image}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        names = sample["name"]
        ids = sample["categoryid"]
        images = sample["image"]

        for i, image in enumerate(images):
            if isinstance(self.output_size, int):
                new_w = self.output_size
                new_h = self.output_size
            else:
                new_w, new_h = self.output_size

            images[i] = cv2.resize(image, (new_w, new_h))

        return {"name": names, "categoryid": ids, "image": images}


class ToTensor(object):
    def __call__(self, sample):
        names = sample["name"]
        ids = sample["categoryid"]
        images = sample["image"]

        names = torch.from_numpy(np.array(names))
        # names = names.type(torch.FloatTensor)

        ids = torch.from_numpy(ids)
        # ids = ids.type(torch.FloatTensor)

        nd_images = np.zeros((8, 3, 400, 400), dtype=int)

        for i, image in enumerate(images):
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)
            # image = image.type(torch.FloatTensor)
            nd_images[i] = image

        images = torch.from_numpy(nd_images)

        return {"name": names, "categoryid": ids, "image": images}


def main():
    dataset = PolyvoreDataset("polyvore-dataset/train_no_dup.json", transform=transforms.Compose([
        Resize(400), ToTensor()
    ]))

    loader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=2)

    batch = next(iter(loader))
    for item in batch["image"]:
        print(item.shape)
        for i, img in enumerate(item):
            print(i, img)
            plt.imshow(np.transpose(img, (1, 2, 0)))
            plt.show()


if __name__ == "__main__":
    main()
