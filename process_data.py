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


def split_train_valid_test(data, test_size=0.3, valid_size=0.3):
    """
    This function is used to split the data into a training, validation and test set.
    :param data: Data to be split.
    :param test_size: Size of the test set relative to all data.
    :param valid_size: Size of the validation set relative to train set.
    :return: Returns training, validation and test sets.
    """
    train, test = train_test_split(data, test_size=test_size)
    train, valid = train_test_split(train, test_size=valid_size)
    return train, valid, test


def main(parse_args):
    products = pd.read_parquet(f"{parse_args.dataset}/products.parquet", engine="pyarrow")
    train_products, valid_products, test_products = split_train_valid_test(products)
    if parse_args.debug:
        print(train_products.sample(1).iloc[0])
        print("Train / valid / test shapes:", train_products.shape, valid_products.shape, test_products.shape)

    unique_name = f"_{int(time.time())}" if parse_args.unique_name else ""
    if parse_args.debug: print(unique_name)

    if not parse_args.debug or args.unique_name:
        train_products.to_parquet(f"{parse_args.target}/products_train{unique_name}.parquet")
        valid_products.to_parquet(f"{parse_args.target}/products_valid{unique_name}.parquet")
        test_products.to_parquet(f"{parse_args.target}/products_test{unique_name}.parquet")

    outfits = pd.read_parquet(f"{parse_args.dataset}/manual_outfits.parquet", engine="pyarrow")
    outfits = add_fitb_queries(outfits, products["product_id"])
    if parse_args.debug: print(outfits.sample(1).iloc[0])

    if not parse_args.debug or args.unique_name:
        outfits.to_parquet(f"{parse_args.target}/outfits{unique_name}.parquet")


if __name__ == "__main__":
    import argparse, pathlib

    parser = argparse.ArgumentParser(description="Split the dataset in training, validation and testing sets and "
                                                 "generate fill-in-the-blank queries from the manual outfits.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dataset", type=pathlib.Path, default="./dataset",
                        help="Path of directory where products and manual outfits files are stored.")
    parser.add_argument("--unique-name", action="store_true", help="Store the files generated with unique names.")
    parser.add_argument("-t", "--target", type=pathlib.Path, default=".",
                        help="Path of target directory to store files.")
    parser.add_argument("--debug", action="store_true",
                        help="No files will be stored and additional information is printed used for debugging. "
                             "Files are stored if unique-names flag is set.")
    args = parser.parse_args()
    if args.debug: print(args)
    main(args)
