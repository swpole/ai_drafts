"""
Auto-generated ComfyUI workflow Python script
Generated from Export (API) JSON format

Коррекция: убран параметр video-preview="" в save_video, так как он не поддерживается в текущей версии ComfyUI.
"""

from comfy_extras.nodes_custom_sampler import KSamplerSelect
from comfy_extras.nodes_custom_sampler import SamplerCustom
from comfy_extras.nodes_lt import LTXVConditioning
from comfy_extras.nodes_lt import LTXVImgToVideo
from comfy_extras.nodes_lt import LTXVScheduler
from comfy_extras.nodes_video import CreateVideo
from comfy_extras.nodes_video import SaveVideo
from nodes import *
from nodes import CLIPLoader
from nodes import CLIPTextEncode
from nodes import CheckpointLoaderSimple
from nodes import LoadImage
from nodes import VAEDecode
import folder_paths
import os
import sys
import torch

def main():
    """Основная функция выполнения workflow"""

    cliploader_38 = CLIPLoader()
    cliploader_38_output = cliploader_38.load_clip(clip_name="t5xxl_fp16.safetensors", type="ltxv", device="default")
    # Output available as cliploader_38_output
    
    checkpointloadersimple_44 = CheckpointLoaderSimple()
    checkpointloadersimple_44_output = checkpointloadersimple_44.load_checkpoint(ckpt_name="ltx-video-2b-v0.9.safetensors")
    # Output available as checkpointloadersimple_44_output
    
    ksamplerselect_73 = KSamplerSelect()
    ksamplerselect_73_output = ksamplerselect_73.get_sampler(sampler_name="euler")
    # Output available as ksamplerselect_73_output
    
    loadimage_78 = LoadImage()
    loadimage_78_output = loadimage_78.load_image(image="ComfyUI_00090_.png")
    # Output available as loadimage_78_output
    
    cliptextencode_6 = CLIPTextEncode()
    cliptextencode_6_output = cliptextencode_6.encode(text="best quality, 4k, HDR, a tracking shot of a man playing guitar", clip=cliploader_38_output[0])
    # Output available as cliptextencode_6_output
    
    cliptextencode_7 = CLIPTextEncode()
    cliptextencode_7_output = cliptextencode_7.encode(text="low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly", clip=cliploader_38_output[0])
    # Output available as cliptextencode_7_output
    
    ltxvimgtovideo_77 = LTXVImgToVideo()
    ltxvimgtovideo_77_output = ltxvimgtovideo_77.generate(width=768, height=512, length=97, batch_size=1, strength=0.15, positive=cliptextencode_6_output[0], negative=cliptextencode_7_output[0], vae=checkpointloadersimple_44_output[2], image=loadimage_78_output[0])
    # Output available as ltxvimgtovideo_77_output
    
    ltxvconditioning_69 = LTXVConditioning()
    ltxvconditioning_69_output = ltxvconditioning_69.append(frame_rate=25, positive=ltxvimgtovideo_77_output[0], negative=ltxvimgtovideo_77_output[1])
    # Output available as ltxvconditioning_69_output
    
    ltxvscheduler_71 = LTXVScheduler()
    ltxvscheduler_71_output = ltxvscheduler_71.get_sigmas(steps=30, max_shift=2.05, base_shift=0.95, stretch=True, terminal=0.1, latent=ltxvimgtovideo_77_output[2])
    # Output available as ltxvscheduler_71_output
    
    samplercustom_72 = SamplerCustom()
    samplercustom_72_output = samplercustom_72.sample(add_noise=True, noise_seed=1067783101414212, cfg=3, model=checkpointloadersimple_44_output[0], positive=ltxvconditioning_69_output[0], negative=ltxvconditioning_69_output[1], sampler=ksamplerselect_73_output[0], sigmas=ltxvscheduler_71_output[0], latent_image=ltxvimgtovideo_77_output[2])
    # Output available as samplercustom_72_output
    
    vaedecode_8 = VAEDecode()
    vaedecode_8_output = vaedecode_8.decode(samples=samplercustom_72_output[0], vae=checkpointloadersimple_44_output[2])
    # Output available as vaedecode_8_output
    
    createvideo_80 = CreateVideo()
    createvideo_80_output = createvideo_80.create_video(fps=24, images=vaedecode_8_output[0])
    # Output available as createvideo_80_output
    
    savevideo_81 = SaveVideo()
    savevideo_81_output = savevideo_81.save_video(filename_prefix="video/ComfyUI", format="auto", codec="auto", video=createvideo_80_output[0])
    # Output available as savevideo_81_output
    

if __name__ == "__main__":
    main()