import random
from collections import Counter
from itertools import combinations

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def process_part_image(data, preprocess, most_common):
    processed = dict()
    for product_id, _, image_path in tqdm(data.iloc, total=len(data)):
        if product_id not in most_common: continue
        with Image.open(image_path) as img_file:
            image = preprocess(img_file)
            processed[product_id] = image

    return processed


def main(parse_args):
    outfits = pd.read_parquet(f"{parse_args.dataset}/manual_outfits.parquet", engine="pyarrow")
    products = pd.read_parquet(f"{parse_args.dataset}/products.parquet", engine="pyarrow")
    product_ids = products[["product_id"]]

    outfit_pairs = set()

    for prods, _ in outfits.iloc:
        outfit_pairs.update([(x1, x2, 1) for x1, x2 in combinations(prods, 2)])

    used_outfit_pairs = set(random.sample(outfit_pairs, parse_args.size)) if not parse_args.full_size else outfit_pairs

    random_pairs = set()
    product_ids = product_ids.values.flatten()

    while len(random_pairs) != len(used_outfit_pairs):
        s = np.random.choice(product_ids, 2)

        if (s[0], s[1]) not in outfit_pairs and (s[1], s[0]) not in outfit_pairs:
            random_pairs.add((s[0], s[1], 0))

    print(
        f"Lengths:\noutfit_pairs: {len(outfit_pairs)}\tused_outfit_pairs: {len(used_outfit_pairs)}\trandom_pairs: {len(random_pairs)}")

    all_pairs = used_outfit_pairs | random_pairs
    all_ids = [i for i1, i2, _ in all_pairs for i in [i1, i2]]
    print("Number of ids: ", len(set(all_ids)))
    most_common = [item for item, c in Counter(all_ids).most_common(len(set(all_ids)) // 3)]
    print("Most common:", len(most_common), all_ids.count(most_common[0]), all_ids.count(most_common[-1]))

    outfit_train, outfit_test = train_test_split(list(used_outfit_pairs), test_size=0.3)
    random_train, random_test = train_test_split(list(random_pairs), test_size=0.3)

    train_pairs = outfit_train + random_train
    test_pairs = outfit_test + random_test

    print(f"Train length: {len(train_pairs)}\t Test length: {len(test_pairs)}")

    pairs = {"train": train_pairs, "test": test_pairs}
    np.save(f"{parse_args.dataset}/pairs.npy", pairs)

    if parse_args.only_pairs: exit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    products_txt_img = pd.read_parquet(f"{parse_args.dataset}/products_text_image.parquet", engine="pyarrow")

    processed = process_part_image(products_txt_img, preprocess, most_common)

    np.save(f"{parse_args.dataset}/processed_image_part.npy", processed)

    print("Finished processing images")

    processed_text = dict()
    for product_id, text, _ in tqdm(products_txt_img.iloc, total=len(products_txt_img)):
        tokens = clip.tokenize(text, truncate=True).squeeze(0)
        processed_text[product_id] = tokens

    np.save(f"{parse_args.dataset}/processed_text.npy", processed_text)
    del processed_text

    print("Finished processing text")


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=pathlib.Path,
                        help="Directory where products and outfits files are stored.")
    parser.add_argument("-s", "--size", type=int, default=200_000,
                        help="Size of each set of valid and non valid pairs")
    parser.add_argument("--full-size", action="store_true", help="Store all pairs instead of sampling them")
    parser.add_argument("--only-pairs", action="store_true")
    main(parser.parse_args())
