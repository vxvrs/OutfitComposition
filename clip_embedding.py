import clip
import pandas as pd
import torch


class OutfitEmbeddingCLIP:
    def __init__(self, device="cpu"):
        self.model = clip.load("ViT-B/32", device=device, jit=False)


def main(parse_args):
    outfits = pd.read_parquet(f"{parse_args.dataset}/products_text_image.parquet", engine="pyarrow")
    products = pd.read_parquet(f"{parse_args.dataset}/outfits.parquet", engine="pyarrow")
    print(outfits, products, sep='\n')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed = OutfitEmbeddingCLIP(device=device)


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=pathlib.Path, help="Directory containing outfits and products files.")
    main(parser.parse_args())
