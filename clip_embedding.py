import clip
import numpy as np
import pandas as pd
import torch

import data_farfetch


def reciprocal_rank(item, ranking):
    for i, ranked_item in enumerate(ranking):
        if item == ranked_item:
            return 1 / (i + 1)


class OutfitEmbeddingCLIP:
    def __init__(self, products, modal, device="cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)

        self.products = products
        self.dataset = data_farfetch.FarfetchDataset(products, clip.tokenize, self.preprocess)
        self.modal = modal

        self.embedding = dict()

    def get_product(self, product_id):
        return self.products.loc[self.products["product_id"] == product_id]

    def embed(self, product_id):
        _, text, image = self.dataset.get_product(product_id)
        text = text.unsqueeze(0)
        image = image.unsqueeze(0)

        with torch.no_grad():
            text_encoding = self.model.encode_text(text.to(self.device)) if "text" in self.modal else None
            image_encoding = self.model.encode_image(image.to(self.device)) if "image" in self.modal else None

        if text_encoding is not None and image_encoding is not None:
            encoding = torch.cat((text_encoding, image_encoding), 1)
        else:
            encoding = text_encoding if text_encoding is not None else image_encoding

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
            # candidate_embed = np.append(outfit_embed, self.embed(product_id).numpy())
            candidate_embed = np.array(
                [self.embed(product_id).cpu().numpy() for product_id in np.append(incomplete_outfit, product_id)])
            candidate_centroid = candidate_embed.mean(axis=0)

            distance = np.linalg.norm(candidate_centroid - base_centroid)
            distances_candidates[product_id] = distance

        return sorted(distances_candidates.items(), key=lambda item: item[1])


def main(parse_args):
    data_farfetch.dataset_path = parse_args.dataset

    products = pd.read_parquet(f"{parse_args.dataset}/products_text_image.parquet", engine="pyarrow")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed = OutfitEmbeddingCLIP(products, parse_args.modal, device=device)

    outfits = pd.read_parquet(f"{parse_args.dataset}/outfits.parquet", engine="pyarrow")

    missing_product = outfits[["outfit_id", "missing_product"]]
    missing_product.to_csv(f"{parse_args.dataset}/missing_product.csv", index=False)

    predicted_product = pd.DataFrame(columns=["outfit_id", "predicted_product"])

    recip_ranks = list()
    for row in outfits[["outfit_id", "incomplete_outfit", "candidates", "missing_product"]].iloc:
        outfit_id, incomplete_outfit, candidates, missing_product = row
        ranking = embed.fitb(incomplete_outfit, candidates)
        predicted_id, _ = ranking[0]

        new_row = pd.DataFrame.from_dict({"outfit_id": [outfit_id], "predicted_product": [predicted_id]})
        predicted_product = pd.concat([predicted_product, new_row], ignore_index=True)

        r_rank = reciprocal_rank(missing_product, [r for r, _ in ranking])
        recip_ranks.append(r_rank)
        print("Reciprocal rank:", r_rank)

    print("Mean reciprocal rank:", np.mean(recip_ranks))
    predicted_product.to_csv(f"{parse_args.dataset}/predicted_product.csv", index=False)

    """
    Outfit retrieval FITB
    - items without options as outfit embedding to compare with the options of fitb
    - look up outfit embeddings with items from fitb and compare those with options (products have to be already known)
      variation maybe possible with look at similar outfits
    """


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dataset", type=pathlib.Path, help="Directory containing outfits and products files.")
    parser.add_argument("-m", "--modal", choices=["text_image", "text", "image"], default="text_image",
                        help="Modalities to use in the network")
    main(parser.parse_args())
