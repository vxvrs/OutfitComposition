import clip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.utils.data import DataLoader

import data_farfetch


class AutoEncoder(nn.Module):
    def __init__(self, input_size=1024, latent_size=64):
        super(AutoEncoder, self).__init__()

        self.l1 = nn.Linear(input_size, input_size // 2)
        self.l2 = nn.Linear(input_size // 2, input_size // 8)
        self.l3 = nn.Linear(input_size // 8, latent_size)

        self.l4 = nn.Linear(latent_size, input_size // 8)
        self.l5 = nn.Linear(input_size // 8, input_size // 2)
        self.l6 = nn.Linear(input_size // 2, input_size)

    def encoder(self, x):
        x = relu(self.l1(x))
        x = relu(self.l2(x))
        x = relu((self.l3(x)))
        return x

    def decoder(self, x):
        x = relu(self.l4(x))
        x = relu(self.l5(x))
        x = self.l6(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_data(device, batch, clip_model, modal):
    _, text, image = batch

    with torch.no_grad():
        text_embed = clip_model.encode_text(text.to(device)) if "text" in modal else None
        image_embed = clip_model.encode_image(image.to(device)) if "image" in modal else None

    if text_embed is not None and image_embed is not None:
        data = torch.cat((text_embed, image_embed), 1)
    else:
        data = text_embed if text_embed is not None else image_embed

    return data


def main(parse_args):
    data_farfetch.dataset_path = parse_args.dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    train_df = pd.read_parquet(f"{data_farfetch.dataset_path}/products_train_text_image.parquet",
                               engine="pyarrow")
    if parse_args.testing: train_df = train_df.sample(100)
    trainset = data_farfetch.FarfetchDataset(train_df, clip.tokenize, preprocess)
    trainloader = DataLoader(trainset, batch_size=50, shuffle=True)

    valid_df = pd.read_parquet(f"{data_farfetch.dataset_path}/products_test_text_image.parquet",
                               engine="pyarrow")
    if parse_args.testing: valid_df = valid_df.sample(30)
    validset = data_farfetch.FarfetchDataset(valid_df, clip.tokenize, preprocess)
    validloader = DataLoader(validset, batch_size=50, shuffle=False)

    print("Data Loaded!")

    input_size = 1024 if parse_args.modal == "text_image" else 512
    ae_model = AutoEncoder(input_size=input_size, latent_size=input_size // 16)
    ae_model = ae_model.to(device)
    print(parse_args.modal, ae_model)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.001)

    n_epochs = 20
    min_valid_loss = np.inf
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        for batch in trainloader:
            data = get_data(device, batch, clip_model, parse_args.modal).type(torch.FloatTensor)
            data = data.to(device)

            optimizer.zero_grad()
            outputs = ae_model(data)

            loss = criterion(outputs, data)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        valid_loss = 0.0

        with torch.no_grad():
            for batch in validloader:
                data = get_data(device, batch, clip_model, parse_args.modal).type(torch.FloatTensor)
                data = data.to(device)

                outputs = ae_model(data)
                loss = criterion(outputs, data)
                valid_loss += loss.item()

        print(f"Epoch {epoch} \t Training Loss: {train_loss} \t Validation Loss: {valid_loss}")

        if min_valid_loss > valid_loss:
            print(f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})")
            min_valid_loss = valid_loss
            torch.save(ae_model, f"models/ae_model_{parse_args.modal}_{len(trainset)}_{len(validset)}.pt")


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=pathlib.Path,
                        help="Directory where products text image processed files are stored.")
    parser.add_argument("-m", "--modal", choices=["text_image", "text", "image"], default="text_image",
                        help="Modalities to use in the network")
    parser.add_argument("--testing", action="store_true",
                        help="Use a small portion of the dataset for testing purposes.")
    main(parser.parse_args())
