import clip
import numpy as np
import pandas as pd
import torch

import data_farfetch


class OutfitEmbeddingCLIP:
    def __init__(self, products, outfits, modal, device="cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.dataset = data_farfetch.FarfetchDataset(products, clip.tokenize, self.preprocess)
        self.products = products
        self.outfits = outfits
        self.modal = modal

        self.embedding = dict()
        self.setup_embedding()

    def get_product(self, product_id):
        return self.products.loc[self.products["product_id"] == product_id]

    def embed(self, product_id):
        row = self.get_product(product_id)
        _, text, image = self.dataset.get_product(product_id)
        text = text.unsqueeze(0)
        image = image.unsqueeze(0)

        with torch.no_grad():
            text_encoding = self.model.encode_text(text.to(self.device)) if "text" in self.modal else None
            image_encoding = self.model.encode_image(image.to(self.device)) if "image" in self.modal else None

        if text_encoding is not None and image_encoding is not None:
            encoding = torch.cat((text_encoding, image_encoding), 1)
        else:
            encoding = text_encoding if text_encoding is not None else image_encoding

        return encoding.squeeze(0)

    def setup_embedding(self):
        for product_id in self.products["product_id"]:
            self.embedding[product_id] = self.embed(product_id)

        np.save("embedding.npy", self.embedding)


def main(parse_args):
    data_farfetch.dataset_path = parse_args.dataset

    products = pd.read_parquet(f"{parse_args.dataset}/products_text_image.parquet", engine="pyarrow")
    outfits = pd.read_parquet(f"{parse_args.dataset}/outfits.parquet", engine="pyarrow")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed = OutfitEmbeddingCLIP(products, outfits, parse_args.modal, device=device)
    embed.embed(16281736)


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=pathlib.Path, help="Directory containing outfits and products files.")
    parser.add_argument("-m", "--modal", choices=["text_image", "text", "image"], default="text_image",
                        help="Modalities to use in the network")
    main(parser.parse_args())
