import pandas
from torch.utils.data import Dataset


class config:
    image_path = "./dataset"


class FarfetchDataset(Dataset):
    def __init__(self, dataset_path):
        self.df = pandas.read_parquet(f"{dataset_path}/products.parquet", engine="pyarrow")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        print(idx)
        return self.df.loc[idx]


def main():
    dataset = FarfetchDataset("./farfetch-dataset")
    print(dataset[:2])


if __name__ == "__main__":
    main()
