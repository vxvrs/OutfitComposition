import random
from itertools import combinations

import numpy as np
import pandas as pd


def main(parse_args):
    outfits = pd.read_parquet(f"{parse_args.dataset}/manual_outfits.parquet", engine="pyarrow")
    products = pd.read_parquet(f"{parse_args.dataset}/products.parquet", engine="pyarrow")
    product_ids = products[["product_id"]]

    outfit_pairs = set()

    for products, _ in outfits.iloc:
        outfit_pairs.update(list(combinations(products, 2)))

    outfit_pairs_small = random.sample(outfit_pairs, parse_args.size)

    random_pairs = set()
    product_ids = product_ids.values.flatten()

    while len(random_pairs) != parse_args.size:
        s = np.random.choice(product_ids, 2)

        if (s[0], s[1]) not in outfit_pairs and (s[1], s[0]) not in outfit_pairs:
            random_pairs.add((s[0], s[1]))

    print(
        f"Lengths:\noutfit_pairs: {len(outfit_pairs)}\toutfit_pairs_small: {len(outfit_pairs_small)}\trandom_pairs: {len(random_pairs)}")

    pairs = {"outfit": set(outfit_pairs_small), "random": random_pairs}
    np.save(f"{parse_args.dataset}/pairs.npy", pairs)


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument("-s", "--size", type=int, default=200_000)
    main(parser.parse_args())
