import pandas as pd
import torch

"""
Plan de campagne: write a class for embedding. 
All products text and/or images need to be converted for using the encoder (preprocessed).
Function for encoding products in outfit and mean of products is outfit encoding (possibly changed).
Item retrieval can be done by embedding remaining products in outfit (mean) and then making new outfit points with each
possible outfit combination and look at possible outfit compared to other items. 
Outfits embedding is stored in class, function for item retrieval (does not have to be preprocessed if ids are linked).
Results from retrieval are stored to file outside of class.
"""


class OutfitEmbedding:
    def __init__(self, ae_model, products, outfits):
        self.ae_model = ae_model
        self.encoder = ae_model.encoder
        self.products = products
        self.outfits = outfits


def main(parse_args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ae_model = torch.load(parse_args.model, map_location=torch.device(device))
    ae_model.eval()
    print(parse_args.modal, ae_model)

    products = pd.read_parquet(f"{parse_args.dataset}/products_text_image.parquet").sample(4)
    print(products)

    outfits = pd.read_parquet(f"{parse_args.dataset}/outfits.parquet")

    embed = OutfitEmbedding(ae_model, products, outfits)
    print(embed.encoder)


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=pathlib.Path, help="Directory containing outfits and products files.")
    parser.add_argument("model", type=pathlib.Path, help="Model file containing autoencoder")
    parser.add_argument("-m", "--modal", choices=["text_image", "text", "image"], default="text_image",
                        help="Modalities to use in the network")
    main(parser.parse_args())
