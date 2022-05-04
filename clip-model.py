import torch
from torch.utils.data import DataLoader
import clip
from PIL import Image
from data import PolyvoreDataset, ToTensor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    trainset = PolyvoreDataset("polyvore-dataset/train_no_dup.json")

    sample = trainset[0]
    for i, line in enumerate(sample["name"]):
        img = Image.open(sample["image"][i])
        img = preprocess(img).unsqueeze(0)
        token = clip.tokenize(line)

        with torch.no_grad():
            print(model.encode_image(img).shape, model.encode_text(token).shape, sep='\t\t')

            logits_per_image, logits_per_text = model(img, token)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            print("Label probs:", probs)

    # with torch.no_grad():
    #     img_f = model.encode_image(image)
    #     print(img_f.shape)


if __name__ == "__main__":
    main()
