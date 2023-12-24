from typing import Iterable
from torchtyping import TensorType

import torch
import torchvision.transforms.functional as TF
import numpy as np

from diffusers import AutoencoderTiny
from utils import skip_torch_init
from utils import load_or_compute
from dim_reduction import umap_reduce, umap_invert
from diffusion_latent_game import PointRendererImages

from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
import einops as eo

class VectorMapper:
    def __init__(self):
        """
        Stable Diffusion VAE maps to 8x64x64 latent "vector"
        But this isn't actually a vector. This lets us turn it 
        into a bunch of vectors in some re-projectable way
        """
        self.store = None
    
    def forward(self, x : TensorType["b", 8, 64, 64]):
        """
        We're gonna assume you can UMAP each feature separately
        """
        return eo.rearrange(x, 'b c h w -> c b (h w)')
    
    def backward(self, x):
        return eo.rearrange(x, 'c b (h w) -> b c h w', h = 64, w = 64)

class SDVAE:
    def __init__(self, model_id, device = 'cuda'):
        with skip_torch_init():
            self.ae = AutoencoderTiny.from_pretrained(model_id, torch_dtype=torch.float16)
            self.ae = self.ae.to(device)

        self.device = device
    
    @torch.no_grad()
    def __call__(self, images : Iterable[Image.Image]):
        images = [image.convert("RGB").resize((512, 512)) for image in images]
        x = torch.stack([TF.to_tensor(image) for image in images]).to(self.device).half()
        x = self.ae.encode(x).latents
        x = VectorMapper().forward(x)
        x = x.detach().cpu().numpy()
        return x

    @torch.no_grad()
    def decode(self, x):


if __name__ == "__main__":
    data_id = "lambdalabs/pokemon-blip-captions"
    model_id = "madebyollin/taesd"

    ds = load_dataset(data_id)['train']
    ds = list(map(lambda x: x['image'], ds))

    embedder = SDVAE(model_id)

    def load_and_get_embeddings(ds, embedder):
        model_id = "madebyollin/taesd"

        chunk_size = 8
        outputs = []
        for i in tqdm(range(0, len(ds), chunk_size)):
            chunk = ds[i:i+chunk_size]
            embeddings = embedder(chunk)
            outputs.append(embeddings)
        outputs = np.concatenate(outputs, axis = 1)
        return outputs

    def umap_reduce_each_channel(x : TensorType[8, "b", "d"]):
        channels_reduced = []
        for x_i in x:
            channels_reduced.append(umap_reduce(x_i))
        
        channels_reduced = np.stack(channels_reduced)
        return channels_reduced

    embeddings = load_or_compute(
        "sdvae-pokemon-embeddings.pkl",
        lambda: load_and_get_embeddings(ds, embedder)
    )

    points = load_or_compute(
        "sdvae-pokemon-umap.pkl",
        lambda: umap_reduce_each_channel(embeddings)
    )

    def generate_image(point) -> Image.Image:
        # points are [8, N, 2]
        points_mean = points.mean(0) # We will use this to compute distances
        channel_embeddings = []

        for channel in range(len(points)):
            channel_embeddings.append(umap_invert(point, points_mean, embeddings[channel]))
            
        x = np.stack(channel_embeddings)
        x = torch.from_numpy(x) # [8, b, 64 * 64]
        x = VectorMapper().backward(x) # [b, 8, 64, 64]
        x = 





    game = PointRendererImages()
    game.run(points.mean(0), ds, callback = generate_image)
