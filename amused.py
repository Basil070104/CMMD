model_name = "amused/amused-512"

# --------------------------------------- DEFINE PROMPTS ---------------------------------------- #
prompts = [
    "Four cartoon dwarfs playing video games in a living room with soda on the table",
    "A bustling city street at night with neon lights reflecting on wet pavement",
    "A cat sleeping on a laptop keyboard, with a mouse and coffee mug on the desk, in sketch style",
    "A laptop, coffee mug, and notebook on a desk with a task lamp, in minimalist style",
    "A basket of yarn, knitting needles, and a half-finished scarf on a rocking chair, in watercolor style"
]
# ----------------------------------------------------------------------------------------------- #

import time, torch, psutil, threading, matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image, AmusedPipeline
from PIL import Image
import pynvml
import os

# Resource utilization tracking variables
cpu_usage = []
ram_usage = []
gpu_usage = []
vram_usage = []
timestamps = []
prompt_end_times = []
monitoring = True
def monitor_resources():
    global monitoring
    while monitoring:
        current_time = time.time() - start_time
        cpu_usage.append(psutil.cpu_percent())
        ram_usage.append(psutil.virtual_memory().used / (1024 ** 3))  # Convert bytes to GB
        gpu_usage.append(torch.cuda.utilization())
        vram_usage.append(torch.cuda.memory_allocated() / (1024 ** 3))  # Convert bytes to GB
        timestamps.append(current_time)
        time.sleep(0.1)  # Polling rate

def get_baseline_usage():
    return {
        'cpu': psutil.cpu_percent(),
        'ram': psutil.virtual_memory().used / (1024 ** 3),  # Convert bytes to GB
        'gpu': torch.cuda.utilization(),
        'vram': torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    }
baseline_usage = get_baseline_usage()
start_time = time.time()
monitor_thread = threading.Thread(target=monitor_resources)
monitor_thread.start()

# -------------------------------------------- PIPELINE LIBRARY -------------------------------------------- #
if model_name == "CompVis/stable-diffusion-v1-4" or model_name == "stabilityai/stable-diffusion-2-1-base":
    pipe = StableDiffusionPipeline.from_pretrained(model_name)
elif model_name == "stabilityai/sdxl-turbo":
    pipe = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16")
elif model_name == "amused/amused-512":
    pipe = AmusedPipeline.from_pretrained(model_name, variant="fp16", torch_dtype=torch.float16)
else:
    print("No associated pipe does not exist. Please add the associated pipe to the pipeline library!")
    quit()
pipe.to("cuda")
# ---------------------------------------------------------------------------------------------------------- #

pipeline_loading_time = time.time()
target_size = (768, 768)
# Generate images for each prompt
times = []
for i, prompt in enumerate(prompts):
    # Calculate image generation time
    start_prompt_time = time.time()
    # image = pipe(prompt, height=target_size[1], width=target_size[0]).images[0] # use this for models 1, 2, 3
    image = pipe(prompt, generator=torch.manual_seed(0)).images[0] # use this for Amused
    end_prompt_time = time.time()
    elapsed_time = end_prompt_time - start_prompt_time
    times.append(elapsed_time)

    # Save image and clear cache
    filename = f"{model_name.replace('/', '_')}_image_{i+1}.png"
    save_dir = "generated_images"
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    
    
    full_path = os.path.join(save_dir, filename)
    
    image = image.resize(target_size, Image.LANCZOS)
    image.save(full_path) 
    print(f"Time taken for prompt '{prompt}': {elapsed_time:.2f} seconds")
    torch.cuda.empty_cache()
    prompt_end_times.append(end_prompt_time - start_time)


# Stop resource monitoring
monitoring = False
monitor_thread.join()