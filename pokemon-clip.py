from typing import Iterable

from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np


from game import PointRendererImages
from utils import load_or_compute
from dim_reduction import umap_reduce

class CLIPImageEmbedder:
    def __init__(self, model_id):
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)

    @torch.no_grad()
    def __call__(self, images : Iterable[Image.Image]):
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        outputs = self.model.get_image_features(pixel_values = inputs.pixel_values)
        outputs = outputs.detach().cpu().numpy()
        return outputs

if __name__ == "__main__":
    def load_and_get_embeddings():
        data_id = "lambdalabs/pokemon-blip-captions"
        model_id = "openai/clip-vit-base-patch32"

        ds = load_dataset(data_id)['train']
        embedder = CLIPImageEmbedder(model_id)

        ds = list(map(lambda x: x['image'], ds))

        chunk_size = 8
        outputs = []
        for i in tqdm(range(0, len(ds), chunk_size)):
            chunk = ds[i:i+chunk_size]
            embeddings = embedder(chunk)
            outputs.append(embeddings)
        outputs = np.concatenate(outputs, axis = 0)
        return outputs

    outputs = load_or_compute(
        'pokemon-clip-embeddings.pkl', load_and_get_embeddings,
        #always_compute=True
    )

    outputs = load_or_compute(
        "pokemon-clip-umap.pkl", lambda: umap_reduce(outputs),
        #always_compute=True
    )

    data_id = "lambdalabs/pokemon-blip-captions"
    ds = load_dataset(data_id)['train']
    ds = list(map(lambda x: x['image'], ds))

    game = PointRendererImages()
    game.run(outputs, ds)
    


    
