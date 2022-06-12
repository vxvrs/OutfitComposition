# Outfit Composition with Siamese Networks

Developed for the Fashion Outfits Challenge organized by Farfetch as part of the
[2022 SIGIR Workshop On eCommerce](https://sigir-ecom.github.io/).

![Siamese model configuration](images/Siamese-Model.png)

All Python files are implemented with **argparse**. By using `-h` and `--help`  a brief description will be displayed
for each argument.

## Requirements

- **PyTorch**
- **CLIP** ([GitHub Repo](https://github.com/openai/CLIP))
- **Pandas**
- **PyArrow**
- **NumPy**
- **Pillow**
- **scikit-learn**

For versions and more packages used,
see [requirements.txt](https://github.com/vxvrs/OutfitComposition/blob/master/requirements.txt).

---

## Preparing Data

1. Use [data_farfetch.py](https://github.com/vxvrs/OutfitComposition/blob/master/data_farfetch.py) to convert the text
   product data into sentences used for training.
    - **Produces files:** *products_text_image.parquet*
2. Use [process_data.py](https://github.com/vxvrs/OutfitComposition/blob/master/process_data.py) to convert the manual
   outfits data into FITB queries that will be used to evaluate the model.
    - **Produces files:** *outfits.parquet*
3. Use [process_pairs.py](https://github.com/vxvrs/OutfitComposition/blob/master/process_pairs.py) to process the data
   into pairs and preprocess part of the data for faster execution.
    - **Produces files:** *pairs.npy*, *processed_text.npy*, *processed_image_part.npy*

## Usage

- Use [clip_siamese.py](https://github.com/vxvrs/OutfitComposition/blob/master/clip_siamese.py) to train the Siamese
  networks with different modalities.
    - **Files needed:** *pairs.npy*, *processed_text.npy*, *processed_image_part.npy*, *products_text_image.parquet*
- Use [clip_embedding.py](https://github.com/vxvrs/OutfitComposition/blob/master/clip_embedding.py) to use either CLIP
  in a zero-shot setting or a trained encoder to answer FITB queries.
    - **Files needed:** *products_text_image.parquet*, *processed_text.npy*, *processed_image_part.npy*, *
      outfits.parquet*
      (when file with FITB queries not provided)