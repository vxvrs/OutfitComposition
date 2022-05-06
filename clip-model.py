import torch
import pandas as pd
import clip
from PIL import Image
from random import randint
from numpy.random import permutation
from sklearn.model_selection import train_test_split


class Config:
    dataset_path = "./farfetch-dataset"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    products = pd.read_parquet(f"{Config.dataset_path}/products.parquet", engine="pyarrow")
    print(products.columns)

    outfits = pd.read_parquet(f"{Config.dataset_path}/manual_outfits.parquet", engine="pyarrow")
    outfits["products_shuffled"] = outfits.apply(lambda row: permutation(row["products"]).tolist(), axis=1)
    outfits["incomplete_outfit"] = outfits.apply(lambda row: row["products_shuffled"][:-1], axis=1)
    outfits["missing_product"] = outfits.apply(lambda row: row["products_shuffled"][-1], axis=1)
    outfits = outfits.drop("products_shuffled", axis=1)

    train_outfits, test_outfits = train_test_split(outfits, test_size=0.3)
    train_outfits, valid_outfits = train_test_split(train_outfits, test_size=0.3)
    print(train_outfits.shape, valid_outfits.shape, test_outfits.shape)

    # for i, line in enumerate(sample["name"]):
    #     img = Image.open(sample["image"][i])
    #     img = preprocess(img).unsqueeze(0)
    #     token = clip.tokenize(line)
    #
    #     with torch.no_grad():
    #         print(model.encode_image(img).shape, model.encode_text(token).shape, sep='\t\t')
    #
    #         logits_per_image, logits_per_text = model(img, token)
    #         probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    #
    #         print("Label probs:", probs)

    # with torch.no_grad():
    #     img_f = model.encode_image(image)
    #     print(img_f.shape)


if __name__ == "__main__":
    main()
