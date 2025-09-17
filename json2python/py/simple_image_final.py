"""
Auto-generated ComfyUI workflow Python script
Generated from Export (API) JSON format

Коррекция: 
"""

from nodes import *
from nodes import CLIPTextEncode
from nodes import CheckpointLoaderSimple
from nodes import EmptyLatentImage
from nodes import KSampler
from nodes import SaveImage
from nodes import VAEDecode
import folder_paths
import os
import sys
import torch

def main():
    """Основная функция выполнения workflow"""

    checkpointloadersimple_4 = CheckpointLoaderSimple()
    checkpointloadersimple_4_output = checkpointloadersimple_4.load_checkpoint(ckpt_name="v1-5-pruned-emaonly-fp16.safetensors")
    # Output available as checkpointloadersimple_4_output
    
    emptylatentimage_5 = EmptyLatentImage()
    emptylatentimage_5_output = emptylatentimage_5.generate(width=512, height=512, batch_size=1)
    # Output available as emptylatentimage_5_output
    
    cliptextencode_6 = CLIPTextEncode()
    cliptextencode_6_output = cliptextencode_6.encode(text="The city in the night", clip=checkpointloadersimple_4_output[1])
    # Output available as cliptextencode_6_output
    
    cliptextencode_7 = CLIPTextEncode()
    cliptextencode_7_output = cliptextencode_7.encode(text="text, watermark", clip=checkpointloadersimple_4_output[1])
    # Output available as cliptextencode_7_output
    
    ksampler_3 = KSampler()
    ksampler_3_output = ksampler_3.sample(seed=990700609792200, steps=20, cfg=8, sampler_name="euler", scheduler="normal", denoise=1, model=checkpointloadersimple_4_output[0], positive=cliptextencode_6_output[0], negative=cliptextencode_7_output[0], latent_image=emptylatentimage_5_output[0])
    # Output available as ksampler_3_output
    
    vaedecode_8 = VAEDecode()
    vaedecode_8_output = vaedecode_8.decode(samples=ksampler_3_output[0], vae=checkpointloadersimple_4_output[2])
    # Output available as vaedecode_8_output
    
    saveimage_9 = SaveImage()
    saveimage_9_output = saveimage_9.save_images(filename_prefix="ComfyUI", images=vaedecode_8_output[0].detach())
    # Output available as saveimage_9_output
    

if __name__ == "__main__":
    main()