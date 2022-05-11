import clip
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import data_farfetch


class AutoEncoder(nn.Module):
    def __init__(self, input_size=1024, latent_size=64):
        super(AutoEncoder, self).__init__()

        self.fc1 = nn.Linear(input_size, input_size // 2)
        self.fc2 = nn.Linear(input_size // 2, input_size // 8)
        self.fc3 = nn.Linear(input_size // 8, latent_size)

        self.t_fc3 = nn.Linear(latent_size, input_size // 8)
        self.t_fc2 = nn.Linear(input_size // 8, input_size // 2)
        self.t_fc1 = nn.Linear(input_size // 2, input_size)

    def encoder(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu((self.fc3(x)))
        return x

    def decoder(self, x):
        x = F.relu(self.t_fc3(x))
        x = F.relu(self.t_fc2(x))
        x = self.t_fc1(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def main(parse_args):
    data_farfetch.dataset_path = parse_args.dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    train_df = pd.read_parquet(f"{data_farfetch.dataset_path}/products_train.parquet", engine="pyarrow")
    train_df = train_df.apply(lambda row: data_farfetch.text_image_pair(row), axis=1)
    trainset = data_farfetch.FarfetchDataset(train_df, clip.tokenize, preprocess)
    trainloader = DataLoader(trainset, batch_size=10, shuffle=True, num_workers=1)

    input_size = 1024 if parse_args.modal == "text_image" else 512
    ae_model = AutoEncoder(input_size=input_size, latent_size=input_size // 16).to(device)
    print(ae_model)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.001)

    n_epochs = 5
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        for batch in trainloader:
            text, image = batch

            with torch.no_grad():
                text_embed = clip_model.encode_text(text) if "text" in parse_args.modal else None
                image_embed = clip_model.encode_image(image) if "image" in parse_args.modal else None

            if text_embed is not None and image_embed is not None:
                data = torch.cat((text_embed, image_embed), 1)
            else:
                data = text_embed if text_embed is not None else image_embed

            data = data.to(device)

            optimizer.zero_grad()
            outputs = ae_model(data)

            loss = criterion(outputs, data)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            print(f"Loss: {train_loss}")

        print(f"Epoch {epoch} \t Loss: {train_loss}")


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dataset", type=pathlib.Path, default="./dataset",
                        help="Path of directory where products and manual outfits files are stored.")
    parser.add_argument("-m", "--modal", choices=["text_image", "text", "image"], default="text_image",
                        help="Modalities to use in the network")
    main(parser.parse_args())
