import clip
import pandas as pd


def image_text_pair(row):
    text_columns = ["product_family", "product_category", "product_sub_category", "product_gender", "product_main_colour",
                    "product_second_color", "product_brand", "product_materials", "product_short_description",
                    "product_attributes", "product_highlights"]
    image_column = "product_image_path"
    return row[image_column]


def main(parse_args):
    dataset = pd.read_parquet(f"{parse_args.dataset}/products_train_1651863895.parquet", engine="pyarrow")
    print(dataset.iloc[3])
    # print(dataset.apply(lambda row: image_text_pair(row), axis=1).tolist())


if __name__ == "__main__":
    import argparse, pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=pathlib.Path, default="./dataset",
                        help="Path of directory where products and manual outfits files are stored.")
    main(parser.parse_args())
