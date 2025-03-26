from diffusers.models import AutoencoderKL
from diffusers import AutoPipelineForText2Image
import torch
from transformers import CLIPProcessor, CLIPModel

#model_name='SG161222/RealVisXL_V4.0'
model_name='SG161222/RealVisXL_V5.0'

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
pipe = AutoPipelineForText2Image.from_pretrained(
    model_name,
    vae=vae,
    use_safetensors=True,
    add_watermarker=False,
    custom_pipeline="lpw_stable_diffusion_xl"
)

clip_model_id = "openai/clip-vit-base-patch16"  # OpenAI CLIP model
clip_model = CLIPModel.from_pretrained(clip_model_id)
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
