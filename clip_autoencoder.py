import clip
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import data_farfetch


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()


def main(parse_args):
    data_farfetch.dataset_path = parse_args.dataset

    clip_model, preprocess = clip.load("ViT-B/32", jit=False)

    train_df = pd.read_parquet(f"{data_farfetch.dataset_path}/products_train.parquet", engine="pyarrow")
    train_df = train_df.apply(lambda row: data_farfetch.image_text_pair(row), axis=1)
    trainset = data_farfetch.FarfetchDataset(train_df, clip.tokenize, preprocess)
    trainloader = DataLoader(trainset, batch_size=10, shuffle=True, num_workers=1)

    for batch in trainloader:
        text, image = batch
        text_embed = clip_model.encode_text(text)
        image_embed = clip_model.encode_image(image)
        print(torch.cat((text_embed, image_embed), 1).shape)


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=pathlib.Path, default="./dataset",
                        help="Path of directory where products and manual outfits files are stored.")
    main(parser.parse_args())
