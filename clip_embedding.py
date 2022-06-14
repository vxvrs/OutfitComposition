import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import data_farfetch
from clip_siamese import SiameseNetwork


def reciprocal_rank(item, ranking):
    for i, ranked_item in enumerate(ranking):
        if item == ranked_item:
            return 1 / (i + 1)


class OutfitEmbeddingCLIP:
    def __init__(self, products, modal, model: SiameseNetwork = None, device="cpu", processed_text=None,
                 processed_image=None):
        self.device = device
        self.tokenizer = clip.tokenize
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.encoder = model.encoder if model is not None else None

        self.products = products
        self.dataset = data_farfetch.FarfetchDataset(products, clip.tokenize, self.preprocess)
        self.modal = modal
        self.processed_text = processed_text
        self.processed_image = processed_image

        self.embedding = dict()

    def __process_row(self, row):
        product_id, text, image_path = row.iloc[0]
        tokens = self.tokenizer(text, truncate=True).squeeze(0)
        with Image.open(image_path) as img_file:
            image = self.preprocess(img_file)

        return product_id, tokens, image

    def get_product(self, product_id):
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

    def embed(self, product_id):
        _, text, image = self.get_product(product_id)
        text = text.unsqueeze(0)
        image = image.unsqueeze(0)

        with torch.no_grad():
            text_encoding = self.clip_model.encode_text(text.to(self.device)) if "text" in self.modal else None
            image_encoding = self.clip_model.encode_image(image.to(self.device)) if "image" in self.modal else None

        if text_encoding is not None and image_encoding is not None:
            encoding = torch.cat((text_encoding, image_encoding), 1)
        else:
            encoding = text_encoding if text_encoding is not None else image_encoding

        if self.encoder:
            with torch.no_grad():
                encoding = encoding.type(torch.FloatTensor)
                encoding = self.encoder(encoding.to(self.device))

        return encoding.squeeze(0)

    def setup_embedding(self):
        for product_id in self.products["product_id"]:
            self.embedding[product_id] = self.embed(product_id)

        np.save(f"embedding_{self.modal}_{len(self.dataset)}.npy", self.embedding)

    def load_embedding(self, filename):
        self.embedding = np.load(filename, allow_pickle=True)

    def fitb(self, incomplete_outfit, candidates):
        outfit_embed = np.array([self.embed(product_id).cpu().numpy() for product_id in incomplete_outfit])
        base_centroid = outfit_embed.mean(axis=0)

        distances_candidates = dict()
        for product_id in candidates:
            candidate_embed = np.append(outfit_embed, self.embed(product_id).cpu().numpy())
            candidate_centroid = candidate_embed.mean(axis=0)

            distance = np.linalg.norm(candidate_centroid - base_centroid)
            distances_candidates[product_id] = distance

        return sorted(distances_candidates.items(), key=lambda item: item[1])


def main(parse_args):
    data_farfetch.dataset_path = parse_args.dataset

    products = pd.read_parquet(f"{parse_args.dataset}/products_text_image.parquet", engine="pyarrow")
    processed_text = np.load(f"{parse_args.dataset}/processed_text.npy",
                             allow_pickle=True).item() if "text" in parse_args.modal else None
    if processed_text: print("Processed text length:", len(processed_text))

    processed_image_part = np.load(f"{parse_args.dataset}/processed_image_part.npy",
                                   allow_pickle=True).item() if "image" in parse_args.modal else None
    if processed_image_part: print("Processed image length:", len(processed_image_part))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(parse_args.model, map_location=device) if parse_args.model else None
    if model: model.eval()

    embed = OutfitEmbeddingCLIP(products, parse_args.modal, model=model, device=device, processed_text=processed_text,
                                processed_image=processed_image_part)

    print("Everything loaded!")

    predicted_product = pd.DataFrame(columns=["outfit_id", "predicted_product"])
    if parse_args.predict:
        outfits = pd.read_parquet(parse_args.predict, engine="pyarrow")

        row_to_embedded = outfits[["outfit_id", "incomplete_outfit", "candidates"]]
        for row in tqdm(row_to_embedded.iloc, total=len(row_to_embedded)):
            outfit_id, incomplete_outfit, candidates = row
            ranking = embed.fitb(incomplete_outfit, candidates)
            predicted_id, _ = ranking[0]

            new_row = pd.DataFrame.from_dict({"outfit_id": [outfit_id], "predicted_product": [predicted_id]})
            predicted_product = pd.concat([predicted_product, new_row], ignore_index=True)

        clip_name = "_clip" if not model else ""
        predicted_product.to_csv(
            f"{parse_args.dataset}/predicted_product{clip_name}_{parse_args.modal}_{parse_args.predict.stem}.csv",
            index=False)
    else:
        outfits = pd.read_parquet(f"{parse_args.dataset}/outfits.parquet", engine="pyarrow")
        if parse_args.evaluate:
            outfits = pd.read_parquet(f"{parse_args.evaluate}", engine="pyarrow")

        clip_name = "_clip" if not model else ""
        missing_product = outfits[["outfit_id", "missing_product"]]
        missing_product.to_csv(f"{parse_args.dataset}/missing_product{clip_name}_{parse_args.modal}.csv", index=False)

        recip_ranks = list()
        row_to_embedded = outfits[["outfit_id", "incomplete_outfit", "candidates", "missing_product"]]
        for row in tqdm(row_to_embedded.iloc, total=len(row_to_embedded)):
            outfit_id, incomplete_outfit, candidates, missing_product = row
            ranking = embed.fitb(incomplete_outfit, candidates)
            predicted_id, _ = ranking[0]

            new_row = pd.DataFrame.from_dict({"outfit_id": [outfit_id], "predicted_product": [predicted_id]})
            predicted_product = pd.concat([predicted_product, new_row], ignore_index=True)

            r_rank = reciprocal_rank(missing_product, [r for r, _ in ranking])
            recip_ranks.append(r_rank)

        print("Mean reciprocal rank:", np.mean(recip_ranks))
        with open(f"{parse_args.dataset}/mean_recip_rank{clip_name}.txt", 'a+') as rank_file:
            rank_file.write(f"MRR-{parse_args.modal}: {np.mean(recip_ranks)}\n")

        clip_name = "_clip" if model else ""
        predicted_product.to_csv(f"{parse_args.dataset}/predicted_product{clip_name}_{parse_args.modal}.csv",
                                 index=False)


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="Use the encoder (either CLIP in zero-shot setting or provided "
                                                 "model) to answer fill-in-the-blank queries found in the file "
                                                 "outfits.parquet or provided fill-in-the-blank queries.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=pathlib.Path, help="Directory containing outfits and products files.")
    parser.add_argument("-m", "--modal", choices=["text_image", "text", "image"], default="text_image",
                        help="Modalities to use in the network")
    parser.add_argument("--evaluate", type=pathlib.Path, help="Path to parquet file containing fill-in-the-blank "
                                                              "queries with correct solutions to be able to calculate"
                                                              " accuracy and mean reciprocal rank.")
    parser.add_argument("--predict", type=pathlib.Path, help="Path to parquet file with fill-in-the-blank queries to "
                                                             "make predictions on. Otherwise outfits.parquet is used "
                                                             "with correct answers to calculate mean reciprocal rank "
                                                             "and accuracy.")
    parser.add_argument("--model", type=pathlib.Path, help="Model to use during encoding. Otherwise just CLIP "
                                                           "encoding is used without trained encoder.")
    main(parser.parse_args())
