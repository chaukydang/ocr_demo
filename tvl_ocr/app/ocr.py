from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torch
from io import BytesIO
from app.model import model, tokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1)
         for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1]
    )
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(BytesIO(image_file)).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    return torch.stack(pixel_values)

import re

def ocr_image(image_bytes: bytes) -> str:
    image_tensor = load_image(image_file=image_bytes, input_size=448, max_num=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pixel_values = image_tensor.to(torch.bfloat16).to(device)

    prompt = "<image>\nMô tả hình ảnh một cách chi tiết trả về dạng markdown."
    generation_config = {
        "max_new_tokens": 512,
        "do_sample": False,
        "num_beams": 3,
        "repetition_penalty": 3.5
    }
    return model.chat(tokenizer, pixel_values, prompt, generation_config)

def ocr_image_and_extract_fields(image_bytes: bytes) -> dict:
    ocr_markdown = ocr_image(image_bytes)

    def extract(pattern):
        match = re.search(pattern, ocr_markdown, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    return {
        "bien_so_xe": extract(r"\*\*Biển số:\*\*\s*([A-Z0-9\-]+)"),
        "ten_tau": extract(r"\*\*Tàu:\*\*\s*([^\n]+)"),
        "so_mooc": extract(r"\*\*Kho/bãi:\*\*\s*([^\n]+)"),
        "can_xe_hang": extract(r"\*\*Cân xe hàng:\*\*\s*([\d\.]+)"),
        "can_xe_rong": extract(r"\*\*Cân xe rỗng:\*\*\s*([\d\.]+)"),
        "trong_luong_hang": extract(r"\*\*Trọng lượng hàng:\*\*\s*([\d\.]+)"),
        "text_raw": ocr_markdown  # optional, dùng để debug
    }
