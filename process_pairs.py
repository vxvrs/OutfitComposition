import random
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main(parse_args):
    outfits = pd.read_parquet(f"{parse_args.dataset}/manual_outfits.parquet", engine="pyarrow")
    products = pd.read_parquet(f"{parse_args.dataset}/products.parquet", engine="pyarrow")
    product_ids = products[["product_id"]]

    outfit_pairs = set()

    for products, _ in outfits.iloc:
        outfit_pairs.update([(x1, x2, 1) for x1, x2 in combinations(products, 2)])

    outfit_pairs_small = random.sample(outfit_pairs, parse_args.size) if not parse_args.full_size else outfit_pairs

    random_pairs = set()
    product_ids = product_ids.values.flatten()

    while len(random_pairs) != len(outfit_pairs_small):
        s = np.random.choice(product_ids, 2)

        if (s[0], s[1]) not in outfit_pairs and (s[1], s[0]) not in outfit_pairs:
            random_pairs.add((s[0], s[1], 0))

    print(
        f"Lengths:\noutfit_pairs: {len(outfit_pairs)}\toutfit_pairs_small: {len(outfit_pairs_small)}\trandom_pairs: {len(random_pairs)}")

    outfit_train, outfit_test = train_test_split(list(outfit_pairs_small), test_size=0.3)
    random_train, random_test = train_test_split(list(random_pairs), test_size=0.3)

    print(f"Train length: {len(outfit_train + random_train)}\t Test length: {len(outfit_test + random_test)}")

    pairs = {"train": outfit_train + random_train, "test": outfit_test + random_test}
    np.save(f"{parse_args.dataset}/pairs.npy", pairs)


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=pathlib.Path,
                        help="Directory where products and outfits files are stored.")
    parser.add_argument("-s", "--size", type=int, default=200_000,
                        help="Size of each set of valid and non valid pairs")
    parser.add_argument("--full-size", action="store_true", help="Store all pairs instead of sampling them")
    main(parser.parse_args())
