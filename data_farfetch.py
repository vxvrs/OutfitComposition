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

import clip
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

dataset_path: str


def text_image_pair(row):
    """
    This function is used to combine all textual data into one full sentence of a given row from a dataframe containing
    Farfetch products. This new sentence is returned with the image path.
    :param row: Row in pandas.DataFrame to process that contains data of a garment.
    :return: Returns combined item description of garment and relative path of the image as a pandas.Series.
    """

    materials = ' '.join(row["product_materials"]) if row["product_materials"] is not None else ""

    highlights = str(row["product_highlights"]).strip('][').replace(',', '') if row["product_highlights"] is not None else ""

    attributes = row["product_attributes"]
    if attributes is not None:
        remove_chars = ['"', '[', ']', '{', '}', ':', "attribute_name", "attribute_values"]
        for char in remove_chars:
            attributes = attributes.replace(char, '')
        attributes = attributes.replace(',', ' ')
    else:
        attributes = ""

    text_columns = ["product_family", "product_category", "product_sub_category", "product_gender",
                    "product_main_colour", "product_second_color", "product_brand", "product_short_description"]
    text = [word for word in row[text_columns] if word != "N/D" and word is not None]

    text = ' '.join([*text, materials, attributes, highlights]).lower()
    non_empty_words = [word for word in text.split(' ') if len(word) > 1]
    text = ' '.join(non_empty_words)

    image_path = f"{dataset_path}/images/{row['product_image_path']}"

    return pd.Series([row["product_id"], text, image_path], index=["product_id", "item_description", "image_path"])


class FarfetchDataset(Dataset):
    """
    Dataset that can be used in combination with Dataloader to load batches.
    The text is tokenized and images are loaded and preprocessed when needed.
    The text is truncated to fit the CLIP model and the data is returned as tensors.
    """

    def __init__(self, dataframe, tokenizer, preprocess, truncate=True):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.truncate = truncate

    def __len__(self):
        return len(self.dataframe)

    def __process_row(self, product_id, text, image_path):
        tokens = self.tokenizer(text, truncate=self.truncate).squeeze(0)
        with Image.open(image_path) as image:
            processed_image = self.preprocess(image)

        return product_id, tokens, processed_image

    def __getitem__(self, idx):
        product_id, text, image_path = self.dataframe.iloc[idx]
        _, text, image_path = self.__process_row(product_id, text, image_path)
        return text, image_path

    def get_product(self, product_id):
        row = self.dataframe.loc[self.dataframe["product_id"] == product_id]
        product_id, text, image_path = row.iloc[0]
        return self.__process_row(product_id, text, image_path)


def main(parse_args):
    global dataset_path
    dataset_path = parse_args.product_file.parents[0]

    _, preprocess = clip.load("ViT-B/32", jit=False)

    dataset = pd.read_parquet(parse_args.product_file, engine="pyarrow")
    if not parse_args.save: dataset = dataset.sample(20)
    text_image = dataset.apply(lambda row: text_image_pair(row), axis=1)
    if parse_args.save:
        filename = f"{dataset_path}/{parse_args.product_file.stem}_text_image.parquet"
        text_image.to_parquet(filename)

    test = FarfetchDataset(text_image, clip.tokenize, preprocess)
    loader = DataLoader(test, batch_size=5, shuffle=True)
    _, text_batch, image_batch = next(iter(loader))
    print(f"Text: {text_batch.shape}", f"Image: {image_batch.shape}", sep='\n')


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="Shows how to load a batch using the DataLoader with farfetch a "
                                                 "Dataset. The text and image sizes from the batch are printed and "
                                                 "the text image pairs can also be stored.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("product_file", type=pathlib.Path, help="File that contains the products you want use. Make "
                                                                "sure the files are in dataset directory.")
    parser.add_argument("--save", action="store_true",
                        help="Save parquet file of text image pairs in same directory as the product file.")
    main(parser.parse_args())
