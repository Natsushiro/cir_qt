from mmcam import get_masking
from model.clip import _transform, load
from open_clip import get_tokenizer
from params import parse_args
from PIL import Image
import torch
from torchvision.transforms import ToPILImage, Normalize, Compose
from utils import is_master, convert_models_to_fp32

args = parse_args()
tokenizer = get_tokenizer("ViT-B-32")

def inverse_transform(tensor):
    denormalize = Normalize(mean=[-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711],
                            std=[1/0.26862954, 1/0.26130258, 1/0.27577711])

    to_pil = ToPILImage()
    inverse_transform = Compose([
        denormalize,
        to_pil,
    ])
    pil_image = inverse_transform(tensor)
    return pil_image


cam_model, _, preprocess_val = load(
           "ViT-B/32",
            jit=False)

convert_models_to_fp32(cam_model)

mask = get_masking(args, model=cam_model)

#image1 = Image.open("./data/000009928.jpg").convert("RGB")
image1 = Image.open("./sample/DSCF2878.JPG").convert("RGB")
text1 = "old itchy monkey in the forest"
image2 = Image.open("./data/000007070.jpg").convert("RGB")
text2= "penguins on a pebble beach"

image1 = preprocess_val(image1)
image2 = preprocess_val(image2)

images = torch.stack((image1, image2)).cuda()

word1 = "station"
word2 = "penguin"
words = [word1, word2]
words = tokenizer(words).cuda()

masked_images = mask(images, words)

sa = 0
for sample in masked_images:
    image = inverse_transform(sample)
    image.save("sample/test%s_2.png" % str(sa))
    sa += 1