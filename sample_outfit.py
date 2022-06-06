import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

plt.interactive(False)

outfits = pd.read_parquet("farfetch-dataset/outfits.parquet", engine="pyarrow")
products = pd.read_parquet("farfetch-dataset/products.parquet", engine="pyarrow")
print(products.sample(1).iloc[0])

sample_outfit = outfits.sample(1).iloc[0]
while len(sample_outfit.products) != 4: sample_outfit = outfits.sample(1).iloc[0]
print(sample_outfit)

# Good outfits: 203903, 46266, 32239
sample_outfit = outfits.loc[outfits.outfit_id == 46266].iloc[-1]
print(sample_outfit)

outfit_pids = sample_outfit.products

for pid in outfit_pids:
    product = products.loc[products.product_id == pid].iloc[0]
    print("farfetch-dataset/images/" + product.product_image_path)

    with Image.open(f"farfetch-dataset/images/{product.product_image_path}") as image:
        plt.imshow(image)
        plt.show()
