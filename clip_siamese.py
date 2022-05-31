import sys

import clip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.functional import relu
from torch.utils.data import Dataset, DataLoader


class ProductPairs(Dataset):
    # TODO: Remove clip tokenizing, preprocessing and use product_ids dictionary for lookup.
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
            (image1, image2)), torch.tensor([label]).type(torch.FloatTensor)


class SiameseNetwork(nn.Module):
    # TODO: Remove "funnel" for use of contrastive loss
    def __init__(self, input_size=1024):
        super(SiameseNetwork, self).__init__()

        self.l1 = nn.Linear(input_size, input_size // 2)
        self.l2 = nn.Linear(input_size // 2, input_size // 4)
        self.l3 = nn.Linear(input_size // 4, input_size // 8)

        concat_size = input_size * 2 // 8
        self.l4 = nn.Linear(concat_size, concat_size // 2)
        self.l5 = nn.Linear(concat_size // 2, concat_size // 4)
        self.l6 = nn.Linear(concat_size // 4, concat_size // 8)
        self.out = nn.Linear(concat_size // 8, 1)

    def encoder(self, x):
        x = relu(self.l1(x))
        x = relu(self.l2(x))
        x = self.l3(x)
        return x

    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        x = torch.cat((x1, x2), 1)
        x = relu(self.l4(x))
        x = relu(self.l5(x))
        x = relu(self.l6(x))
        x = torch.sigmoid(self.out(x))

        return x


def get_data(device, clip_model, modal, text, image):
    with torch.no_grad():
        text_embed = clip_model.encode_text(text.to(device)) if "text" in modal else None
        image_embed = clip_model.encode_image(image.to(device)) if "image" in modal else None

    if text_embed is not None and image_embed is not None:
        data = torch.cat((text_embed, image_embed), 1)
    else:
        data = text_embed if text_embed is not None else image_embed

    data = data.type(torch.FloatTensor)

    return data.to(device)


def main(parse_args):
    log_file = open(f"{parse_args.modal}.log", 'w')
    sys.stdout = log_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    pairs = np.load(f"{parse_args.dataset}/pairs.npy", allow_pickle=True)
    pairs: dict = pairs.item()
    print(len(pairs["train"]), len(pairs["test"]))

    products = pd.read_parquet(f"{parse_args.dataset}/products_text_image.parquet", engine="pyarrow")
    train_set = ProductPairs(products, pairs["train"], clip.tokenize, preprocess)
    valid_set = ProductPairs(products, pairs["test"], clip.tokenize, preprocess)

    train_loader = DataLoader(train_set, batch_size=10_000, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=10_000, shuffle=False)

    input_size = 1024 if parse_args.modal == "text_image" else 512
    model = SiameseNetwork(input_size)
    model = model.to(device)
    print(parse_args.modal, model)
    # TODO: Implement and use contrastive loss instead of cross entropy.
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    min_valid_loss = np.inf
    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        for batch in train_loader:
            pid, text, image, label = batch
            label = label.to(device)

            text1, text2 = torch.unbind(text, 1)
            image1, image2 = torch.unbind(image, 1)

            data1 = get_data(device, clip_model, parse_args.modal, text1, image1)
            data2 = get_data(device, clip_model, parse_args.modal, text2, image2)

            optimizer.zero_grad()
            output = model(data1, data2)

            loss = criterion(output, label)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                pid, text, image, label = batch
                label = label.to(device)

                text1, text2 = torch.unbind(text, 1)
                image1, image2 = torch.unbind(image, 1)

                data1 = get_data(device, clip_model, parse_args.modal, text1, image1)
                data2 = get_data(device, clip_model, parse_args.modal, text2, image2)

                output = model(data1, data2)
                loss = criterion(output, label)
                valid_loss += loss.item()

        print(f"Epoch {epoch} \t Training Loss: {train_loss} \t Validation Loss: {valid_loss}")

        if min_valid_loss > valid_loss:
            print(f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})")
            min_valid_loss = valid_loss
            torch.save(model, f"models/sm_model_{parse_args.modal}_e{epoch}_{len(train_set)}_{len(valid_set)}.pt")

    log_file.close()


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=pathlib.Path,
                        help="Directory where products and pairs files are stored.")
    parser.add_argument("-m", "--modal", choices=["text_image", "text", "image"], default="text_image",
                        help="Modalities to use in the network")
    main(parser.parse_args())
