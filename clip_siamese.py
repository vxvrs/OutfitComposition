import clip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.functional import relu, pairwise_distance
from torch.utils.data import Dataset, DataLoader


class ProductPairs(Dataset):
    def __init__(self, products, pairs, tokenizer, preprocess, modal, processed_text=None, processed_image=None):
        self.products = products
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.modal = modal
        self.processed_text = processed_text
        self.processed_image = processed_image

    def __len__(self):
        return len(self.pairs)

    def __process_row(self, row):
        product_id, text, image_path = row.iloc[0]
        tokens = self.tokenizer(text, truncate=True).squeeze(0)
        with Image.open(image_path) as img_file:
            image = self.preprocess(img_file)

        return product_id, tokens, image

    def __load_product(self, product_id):
        text = torch.empty(77)
        image = torch.empty((3, 224, 224))

        if self.processed_image and product_id in self.processed_image.keys():
            image = self.processed_image[product_id]
        elif "image" in self.modal:
            row = self.products.loc[self.products.product_id == product_id]
            product_id, text, image = self.__process_row(row)

        if self.processed_text and product_id in self.processed_text.keys():
            text = self.processed_text[product_id]
        elif "text" in self.modal:
            row = self.products.loc[self.products.product_id == product_id]
            product_id, text, image = self.__process_row(row)

        return product_id, text, image

    def __getitem__(self, idx):
        product_id1, product_id2, label = self.pairs[idx]

        product_id1, text1, image1 = self.__load_product(product_id1)
        product_id2, text2, image2 = self.__load_product(product_id2)

        return torch.tensor([product_id1, product_id2]), torch.stack((text1, text2)), torch.stack(
            (image1, image2)), torch.tensor([label])


class SiameseNetwork(nn.Module):
    def __init__(self, input_size=1024):
        super(SiameseNetwork, self).__init__()

        self.l1 = nn.Linear(input_size, input_size // 2)
        self.l2 = nn.Linear(input_size // 2, input_size // 4)
        self.l3 = nn.Linear(input_size // 4, input_size // 8)
        self.l4 = nn.Linear(input_size // 8, input_size // 16)

    def encoder(self, x):
        x = relu(self.l1(x))
        x = relu(self.l2(x))
        x = relu(self.l3(x))
        x = self.l4(x)
        return x

    def forward(self, x1, x2):
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        return x1, x2


class ContrastiveLoss(torch.nn.Module):
    """
    Credits to: James D. McCaffrey
    (https://jamesmccaffrey.wordpress.com/2022/03/17/yet-another-siamese-neural-network-example-using-pytorch/)
    """

    def __init__(self, m=2.0):
        super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
        self.m = m  # margin or radius

    def forward(self, y1, y2, flag):
        # flag = 1 means y1 and y2 are supposed to be same
        # flag = 0 means y1 and y2 are supposed to be different

        euc_dist = pairwise_distance(y1, y2)

        loss = torch.mean(flag * torch.pow(euc_dist, 2) +
                          (1 - flag) * torch.pow(torch.clamp(self.m - euc_dist, min=0.0), 2))

        return loss


def clip_encode(device, clip_model, modal, text, image):
    with torch.no_grad():
        text_embed = clip_model.encode_text(text.to(device)) if "text" in modal else None
        image_embed = clip_model.encode_image(image.to(device)) if "image" in modal else None

    if text_embed is not None and image_embed is not None:
        data = torch.cat((text_embed, image_embed), 1)
    else:
        data = text_embed if text_embed is not None else image_embed

    data = data.type(torch.FloatTensor)

    return data.to(device)


def process_batch(batch, device, clip_model, modal):
    pid, text, image, label = batch
    label = label.to(device)

    text1, text2 = torch.unbind(text, 1)
    image1, image2 = torch.unbind(image, 1)

    data1 = clip_encode(device, clip_model, modal, text1, image1)
    data2 = clip_encode(device, clip_model, modal, text2, image2)

    return data1, data2, label


def main(parse_args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    pairs = np.load(f"{parse_args.dataset}/pairs.npy", allow_pickle=True)
    pairs: dict = pairs.item()
    print("Pairs length:", len(pairs["train"]), len(pairs["test"]))

    processed_text = np.load(f"{parse_args.dataset}/processed_text.npy",
                             allow_pickle=True).item() if "text" in parse_args.modal else None
    if processed_text: print("Processed text length:", len(processed_text))

    processed_image_part = np.load(f"{parse_args.dataset}/processed_image_part_20.npy",
                                   allow_pickle=True).item() if "image" in parse_args.modal else None
    if processed_image_part: print("Processed image length:", len(processed_image_part))

    products = pd.read_parquet(f"{parse_args.dataset}/products_text_image.parquet", engine="pyarrow")
    train_set = ProductPairs(products, pairs["train"], clip.tokenize, preprocess, parse_args.modal,
                             processed_text=processed_text, processed_image=processed_image_part)
    valid_set = ProductPairs(products, pairs["test"], clip.tokenize, preprocess, parse_args.modal)

    train_loader = DataLoader(train_set, batch_size=10_000, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=10_000, shuffle=False)

    print("All data loaded!")

    input_size = 1024 if parse_args.modal == "text_image" else 512
    model = SiameseNetwork(input_size)
    model = model.to(device)
    print(parse_args.modal, model)
    criterion = ContrastiveLoss()
    print(criterion)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    min_valid_loss = np.inf
    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        for batch in train_loader:
            data1, data2, label = process_batch(batch, device, clip_model, parse_args.modal)

            optimizer.zero_grad()
            y1, y2 = model(data1, data2)

            loss = criterion(y1, y2, label)
            print(loss)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                data1, data2, label = process_batch(batch, device, clip_model, parse_args.modal)

                output = model(data1, data2)
                loss = criterion(*output, label)
                valid_loss += loss.item()

        print(f"Epoch {epoch} \t Training Loss: {train_loss} \t Validation Loss: {valid_loss}")

        if min_valid_loss > valid_loss:
            print(f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})")
            min_valid_loss = valid_loss
            torch.save(model, f"models/sm_model_{parse_args.modal}_e{epoch}_{len(train_set)}_{len(valid_set)}.pt")


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=pathlib.Path,
                        help="Directory where products and pairs files are stored.")
    parser.add_argument("-m", "--modal", choices=["text_image", "text", "image"], default="text_image",
                        help="Modalities to use in the network.")
    main(parser.parse_args())
