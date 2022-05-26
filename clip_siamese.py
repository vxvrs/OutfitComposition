import clip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.functional import relu
from torch.utils.data import Dataset, DataLoader


class ProductPairs(Dataset):
    def __init__(self, products, pairs, tokenizer, preprocess):
        self.products = products
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.preprocess = preprocess

    def __len__(self):
        return len(self.pairs)

    def __process_row(self, row):
        product_id, text, image_path = row.iloc[0]
        tokens = self.tokenizer(text, truncate=True).squeeze(0)
        with Image.open(image_path) as img_file:
            image = self.preprocess(img_file)

        return product_id, tokens, image

    def __getitem__(self, idx):
        product_id1, product_id2, label = self.pairs[idx]

        row1 = self.products.loc[self.products.product_id == product_id1]
        row2 = self.products.loc[self.products.product_id == product_id2]

        product_id1, text1, image1 = self.__process_row(row1)
        product_id2, text2, image2 = self.__process_row(row2)

        return torch.tensor([product_id1, product_id2]), torch.stack((text1, text2)), torch.stack(
            (image1, image2)), label


class SiameseNetwork(nn.Module):
    def __init__(self, input_size=1024):
        super(SiameseNetwork, self).__init__()

        self.l1 = nn.Linear(input_size, input_size // 2)
        self.l2 = nn.Linear(input_size // 2, input_size // 4)
        self.l3 = nn.Linear(input_size // 4, input_size // 8)
        self.out = nn.Linear(input_size // 8, 1)

    def single_forward(self, x):
        x = relu(self.l1(x))
        x = relu(self.l2(x))
        x = relu(self.l3(x))
        x = torch.sigmoid(x)
        return x

    def forward(self, x1, x2):
        x1 = self.single_forward(x1)
        x2 = self.single_forward(x2)
        return x1, x2


def main(parse_args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    pairs = np.load(f"{parse_args.dataset}/pairs.npy", allow_pickle=True).item()
    # all_pairs = list(pairs["outfit"] | pairs["random"])
    all_pairs = list()
    for outfit, random in zip(pairs["outfit"], pairs["random"]):
        all_pairs.append((outfit[0], outfit[1], 1))
        all_pairs.append((random[0], random[1], 0))

    products = pd.read_parquet(f"{parse_args.dataset}/products_text_image.parquet", engine="pyarrow")
    dataset = ProductPairs(products, all_pairs, clip.tokenize, preprocess)

    loader = DataLoader(dataset, batch_size=5, shuffle=True)

    model = SiameseNetwork()
    print(model)

    # for batch in loader:
    #     pid, txt, img, lbl = batch
    #     print(pid, lbl)
    #     break


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=pathlib.Path)
    main(parser.parse_args())
