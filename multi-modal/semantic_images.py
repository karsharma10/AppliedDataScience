import os

import torch
import clip
from PIL import Image


class OpenAICLip:

    def __init__(self, clip_model_name: str, images_file_path: str):
        """
        Initializes our clip class
        :param clip_model_name:
        :param images_file_path:
        """
        self.clip_model_name = clip_model_name
        self.images_file_path = images_file_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None

    def _load_clip_model(self):
        self.model, self.preprocess = clip.load(self.clip_model_name, device=self.device)

    def load_image_embeddings(self):
        images = os.listdir(self.images_file_path)
        for image in images:
            image_path = os.path.join(self.images_file_path, image)
            image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image)

            print("Image features shape:", image_features.shape)

    def compute(self):
        self._load_clip_model()

        assert (self.preprocess and self.model), "Need to initialize Model and Preprocessor"


        image = self.preprocess(Image.open("cat.png")).unsqueeze(0).to(self.device)
        text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        print("Label probs:", probs)




if __name__ == "__main__":
    o = OpenAICLip("ViT-B/32", None)
    o.compute()



