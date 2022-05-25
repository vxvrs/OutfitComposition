import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ProductPairs(Dataset):
    def __init__(self, products, pairs, tokenizer, preprocess):
        self.products = products
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.preprocess = preprocess

    def __len__(self):
        return len(self.pairs)

    def __process_text_image(self, text, image_path):
        tokens = self.tokenizer(text).squeeze(0)
        with Image.open(image_path) as img_file:
            image = self.preprocess(img_file)

        print(tokens)
        return tokens, image

    def __getitem__(self, idx):
        product_id1, product_id2 = self.pairs[idx]
        print(product_id1, product_id2)

        row1 = self.products.loc[self.products.product_id == product_id1]
        row2 = self.products.loc[self.products.product_id == product_id2]
        print(row1, row2, sep='\n')


def main(parse_args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    pairs = np.load(f"{parse_args.dataset}/pairs.npy", allow_pickle=True).item()
    all_pairs = list(pairs["outfit"] | pairs["random"])

    products = pd.read_parquet(f"{parse_args.dataset}/products_text_image.parquet", engine="pyarrow")
    dataset = ProductPairs(products, all_pairs, clip.tokenize, preprocess)
    print(dataset[0])


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=pathlib.Path)
    main(parser.parse_args())
