import clip
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

dataset_path: str


def image_text_pair(row):
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

    return pd.Series([text, image_path], index=["item_description", "image_path"])


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

    def __getitem__(self, idx):
        text, image_path = self.dataframe.iloc[idx]

        tokens = self.tokenizer(text, truncate=self.truncate).squeeze(0)
        with Image.open(image_path) as image:
            processed_image = self.preprocess(image)

        return tokens, processed_image


def main(parse_args):
    global dataset_path
    dataset_path = parse_args.dataset

    _, preprocess = clip.load("ViT-B/32", jit=False)

    dataset = pd.read_parquet(f"{parse_args.dataset}/products_train.parquet", engine="pyarrow")
    text_image = dataset.iloc[:100].apply(lambda row: image_text_pair(row), axis=1)

    test = FarfetchDataset(text_image, clip.tokenize, preprocess)
    loader = DataLoader(test, batch_size=5, shuffle=True)
    text_batch, image_batch = next(iter(loader))
    print(f"Text: {text_batch.shape}", f"Image: {image_batch.shape}", sep='\n')


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="Shows how to load a batch using the DataLoader with farfetch a "
                                                 "Dataset. The text and image sizes from the batch are printed.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dataset", type=pathlib.Path, default="./dataset",
                        help="Path of directory where products and manual outfits files are stored.")
    main(parser.parse_args())
