import json

import cv2
import numpy as np
from torch.utils.data import Dataset


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


def load_image(filename):
    img = cv2.imread(filename)
    return img[..., ::-1]
    # img = np.transpose(img, (2, 0, 1))
    # print(img.shape)
    # img_tensor = torch.from_numpy(img)
    # img_tensor = img_tensor.type("torch.FloatTensor")
    # img_tensor *= 1/255
    # img_tensor = img_tensor.unsqueeze(0)
    # print(img_tensor.shape)
    # return img_tensor


def load_data(filename):
    json_data = load_json(filename)

    filtered_data = list()
    for data in json_data:
        outfit = dict([(key, list()) for key in ["name", "categoryid", "image_filename"]])

        for item in data["items"]:
            if item["name"] != "polyvore" and item["name"] != "":
                outfit["name"].append(item["name"])
                outfit["categoryid"].append(item["categoryid"])
                image_filename = f"polyvore-dataset/images/{data['set_id']}/{item['index']}.jpg"
                outfit["image_filename"].append(image_filename)

        outfit["name"] = np.array(outfit["name"])
        outfit["categoryid"] = np.array(outfit["categoryid"])
        outfit["image_filename"] = np.array(outfit["image_filename"])
        filtered_data.append(outfit)

    return np.array(filtered_data)


class PolyvoreDataset(Dataset):
    def __init__(self, json_filename, transform=None):
        self.data = load_data(json_filename)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.data[idx]["name"]
        categoryid = self.data[idx]["categoryid"]
        image_filename = self.data[idx]["image_filename"]
        image = [load_image(filename) for filename in image_filename]

        sample = {"name": name, "categoryid": categoryid, "image": image}

        if self.transform:
            sample = self.transform(sample)

        return sample


def main():
    dataset = PolyvoreDataset("polyvore-dataset/train_no_dup.json")
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    #
    # for i, batch in enumerate(dataloader):
    #     print(i, batch)


if __name__ == "__main__":
    main()
