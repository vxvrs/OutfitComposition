import clip
import pandas as pd

import data_farfetch


class OutfitEmbeddingCLIP:
    def __init__(self, products, modal, device="cpu"):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.dataset = data_farfetch.FarfetchDataset(products, clip.tokenize, self.preprocess)
        self.products = products
        self.modal = modal
        self.embedding = dict()

    def get_product(self, product_id):
        return self.products.loc[self.products["product_id"] == product_id]

    def embed(self, product_id):
        row = self.get_product(product_id)
        # print(self.products.index[self.products["product_id"] == product_id].to_list())

        # text_encoding = self.model.encode_text()


def main(parse_args):
    data_farfetch.dataset_path = parse_args.dataset

    products = pd.read_parquet(f"{parse_args.dataset}/products_text_image.parquet", engine="pyarrow")
    outfits = pd.read_parquet(f"{parse_args.dataset}/outfits.parquet", engine="pyarrow")
    print(products, outfits, sep='\n')

    print(products.loc[products["product_id"] == 16281736])
    embed = OutfitEmbeddingCLIP(products, parse_args.modal)
    print(embed.dataset.getdata(16281736))
    embed.embed(16281736)


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=pathlib.Path, help="Directory containing outfits and products files.")
    parser.add_argument("-m", "--modal", choices=["text_image", "text", "image"], default="text_image",
                        help="Modalities to use in the network")
    main(parser.parse_args())
