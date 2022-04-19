import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

def load_json(filename):
    with open(filename) as f:
        return json.load(f)


def read_image(set_id, index):
    img = cv2.imread(f"polyvore-dataset/images/{set_id}/{index}.jpg")
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
        outfit = dict([(key, list()) for key in ["name", "categoryid", "image"]])

        for item in data["items"]:
            if item["name"] != "polyvore":
                outfit["name"].append(item["name"])
                outfit["categoryid"].append(item["categoryid"])
                image = read_image(data["set_id"], item["index"])
                outfit["image"].append(image)

        outfit["name"] = np.array(outfit["name"])
        outfit["categoryid"] = np.array(outfit["categoryid"])
        filtered_data.append(outfit)

    filtered_data = np.array(filtered_data)

    print(filtered_data.shape)


class PolyvoreDataset(Dataset):
    def __init__(self, json_filename):
        self.json_filename = json_filename


def main():
    json_data = load_data("polyvore-dataset/train_no_dup.json")


if __name__ == "__main__":
    main()
