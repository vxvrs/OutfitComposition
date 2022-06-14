"""
Copyright 2022 Viggo Overes

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import time
from random import randint

import pandas as pd
from numpy.random import permutation
from sklearn.model_selection import train_test_split


def add_fitb_queries(outfits, product_ids):
    """
    This function is used to add fill-in-the-blank queries of the outfits to the Dataframe.
    :param outfits: Dataframe contain all manual outfits.
    :param product_ids: All product ids of the items in the dataset.
    :return: Returns Dataframe with columns incomplete_outfit, missing_product and candidates added.
    """
    outfits["products_shuffled"] = outfits.apply(lambda row: permutation(row["products"]).tolist(), axis=1)
    outfits["incomplete_outfit"] = outfits.apply(lambda row: row["products_shuffled"][:-1], axis=1)
    outfits["missing_product"] = outfits.apply(lambda row: row["products_shuffled"][-1], axis=1)

    def candidates(row, min_n=8, max_n=40):
        n = randint(min_n, max_n)
        c = product_ids.sample(n).unique().tolist()
        c.append(row["missing_product"])
        return list(set(c))

    outfits["candidates"] = outfits.apply(lambda row: candidates(row), axis=1)

    return outfits.drop("products_shuffled", axis=1)


def main(parse_args):
    products = pd.read_parquet(f"{parse_args.dataset}/products.parquet", engine="pyarrow")

    train_products, test_products = train_test_split(products, test_size=0.3)
    if parse_args.debug:
        print(train_products.sample(1).iloc[0])
        print("Train / test shapes:", train_products.shape, test_products.shape)

    unique_name = f"_{int(time.time())}" if parse_args.unique_name else ""
    if parse_args.debug: print(unique_name)

    # Not used, possibly remove
    if not parse_args.debug or args.unique_name:
        train_products.to_parquet(f"{parse_args.dataset}/products_train{unique_name}.parquet")
        test_products.to_parquet(f"{parse_args.dataset}/products_test{unique_name}.parquet")

    outfits = pd.read_parquet(f"{parse_args.dataset}/manual_outfits.parquet", engine="pyarrow")
    outfits = add_fitb_queries(outfits, products["product_id"])
    if parse_args.debug: print(outfits.sample(1).iloc[0])

    if not parse_args.debug or args.unique_name:
        outfits.to_parquet(f"{parse_args.dataset}/outfits{unique_name}.parquet")


if __name__ == "__main__":
    import argparse, pathlib

    parser = argparse.ArgumentParser(description="Split the dataset in training and testing sets and "
                                                 "generate fill-in-the-blank queries from the manual outfits.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=pathlib.Path,
                        help="Path of directory where products and manual outfits files are stored. This will also be "
                             "the target directory to store files.")
    parser.add_argument("--unique-name", action="store_true", help="Store the files generated with unique names.")
    parser.add_argument("--debug", action="store_true",
                        help="No files will be stored and additional information is printed used for debugging. "
                             "Files are stored if unique-names flag is set.")
    args = parser.parse_args()
    if args.debug: print(args)
    main(args)
