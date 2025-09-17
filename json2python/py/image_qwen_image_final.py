"""
Auto-generated ComfyUI workflow Python script
Generated from Export (API) JSON format

Коррекция: добавлен обратный штрих в многострочном промпте

"""

from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
from comfy_extras.nodes_sd3 import EmptySD3LatentImage
from nodes import *
from nodes import CLIPLoader
from nodes import CLIPTextEncode
from nodes import KSampler
from nodes import SaveImage
from nodes import UNETLoader
from nodes import VAEDecode
from nodes import VAELoader
import folder_paths
import os
import sys
import torch

def main():
    """Основная функция выполнения workflow"""

    unetloader_37 = UNETLoader()
    unetloader_37_output = unetloader_37.load_unet(unet_name="qwen_image_fp8_e4m3fn.safetensors", weight_dtype="default")
    # Output available as unetloader_37_output
    
    cliploader_38 = CLIPLoader()
    cliploader_38_output = cliploader_38.load_clip(clip_name="qwen_2.5_vl_7b_fp8_scaled.safetensors", type="qwen_image", device="default")
    # Output available as cliploader_38_output
    
    vaeloader_39 = VAELoader()
    vaeloader_39_output = vaeloader_39.load_vae(vae_name="qwen_image_vae.safetensors")
    # Output available as vaeloader_39_output
    
    emptysd3latentimage_58 = EmptySD3LatentImage()
    emptysd3latentimage_58_output = emptysd3latentimage_58.generate(width=1328, height=1328, batch_size=1)
    # Output available as emptysd3latentimage_58_output
    
    modelsamplingauraflow_66 = ModelSamplingAuraFlow()
    modelsamplingauraflow_66_output = modelsamplingauraflow_66.patch_aura(shift=3.1000000000000005, model=unetloader_37_output[0])
    # Output available as modelsamplingauraflow_66_output
    
    cliptextencode_6 = CLIPTextEncode()
    cliptextencode_6_output = cliptextencode_6.encode(text="\"A vibrant, warm neon-lit street scene in Hong Kong at the afternoon, with a mix of colorful Chinese and English signs glowing brightly. The atmosphere is lively, cinematic, and rain-washed with reflections on the pavement. The colors are vivid, full of pink, blue, red, and green hues. Crowded buildings with overlapping neon signs. 1980s Hong Kong style. Signs include:\
\"龍鳳冰室\" \"金華燒臘\" \"HAPPY HAIR\" \"鴻運茶餐廳\" \"EASY BAR\" \"永發魚蛋粉\" \"添記粥麵\" \"SUNSHINE MOTEL\" \"美都餐室\" \"富記糖水\" \"太平館\" \"雅芳髮型屋\" \"STAR KTV\" \"銀河娛樂城\" \"百樂門舞廳\" \"BUBBLE CAFE\" \"萬豪麻雀館\" \"CITY LIGHTS BAR\" \"瑞祥香燭莊\" \"文記文具\" \"GOLDEN JADE HOTEL\" \"LOVELY BEAUTY\" \"合興百貨\" \"興旺電器\" And the background is warm yellow street and with all stores' lights on.", clip=cliploader_38_output[0])
    # Output available as cliptextencode_6_output
    
    cliptextencode_7 = CLIPTextEncode()
    cliptextencode_7_output = cliptextencode_7.encode(text="", clip=cliploader_38_output[0])
    # Output available as cliptextencode_7_output
    
    ksampler_3 = KSampler()
    ksampler_3_output = ksampler_3.sample(seed=795629966006368, steps=20, cfg=2.5, sampler_name="euler", scheduler="simple", denoise=1, model=modelsamplingauraflow_66_output[0], positive=cliptextencode_6_output[0], negative=cliptextencode_7_output[0], latent_image=emptysd3latentimage_58_output[0])
    # Output available as ksampler_3_output
    
    vaedecode_8 = VAEDecode()
    vaedecode_8_output = vaedecode_8.decode(samples=ksampler_3_output[0], vae=vaeloader_39_output[0])
    # Output available as vaedecode_8_output
    
    saveimage_60 = SaveImage()
    saveimage_60_output = saveimage_60.save_images(filename_prefix="ComfyUI", images=vaedecode_8_output[0].detach())
    # Output available as saveimage_60_output
    

if __name__ == "__main__":
    main()