import torch
import pandas
from torch.utils.data import Dataset, DataLoader

class config:
    image_path = "./dataset"


class FarfetchDataset(Dataset):
    def __init__(self, dataset_path):
        self.df = pandas.read_parquet(f"{dataset_path}/products.parquet", engine="pyarrow")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]


def main():
    dataset = FarfetchDataset("./dataset/")
    print(dataset[1000]["product_attributes"])


if __name__ == "__main__":
    main()
