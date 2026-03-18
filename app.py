# PyTorch 2.8 (temporary hack)
import os

os.system('pip install --upgrade --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu126 "torch<2.9" "torchvision" spaces')
os.environ['DIFFUSERS_ENABLE_HUB_KERNELS']='yes'
import torchvision.transforms.functional as TF
# Actual demo code
import gradio as gr
import numpy as np
import spaces

import torch
import random
from PIL import Image

from pipeline import GenSIRR
from diffusers.utils import load_image
import torch.nn.functional as F
from optimization import optimize_pipeline_

MAX_SEED = np.iinfo(np.int32).max
from huggingface_hub import hf_hub_download

def pad_for_model(image: torch.Tensor, multiple: int):
    """Pad the tensor image so height/width are divisible by ``multiple``."""

    height, width = image.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, (0, 0)

    padded = F.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze(0)
    return padded, (pad_h, pad_w)

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a [C,H,W] tensor in [0,1] to a PIL image."""
    # tensor = tensor.clamp(-1.0, 1.0) / 2.0 + 0.5
    tensor = tensor.clamp(0.0, 1.0)
    array = tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array)

def load_deepspeed_weights(model, checkpoint_path) -> None:
    """Load LoRA weights from a DeepSpeed ZeRO Stage 2 checkpoint into the model."""
    tensor_path = checkpoint_path
    # LOGGER.info("Loading ZeRO checkpoint from %s", tensor_path)
    raw_state = torch.load(tensor_path, map_location="cpu")
    module_state: Dict[str, torch.Tensor] = raw_state.get("module")
    if module_state is None:
        raise KeyError("Checkpoint is missing the 'module' state dict")

    # Remove the Lightning prefix so it matches the FluxKontext state dict.
    cleaned_state = {key[len("net_g."):]: value for key, value in module_state.items() if key.startswith("net_g.")}

    missing, unexpected = model.load_state_dict(cleaned_state, strict=True)

pipe = GenSIRR("black-forest-labs/FLUX.1-Kontext-dev")
load_deepspeed_weights(pipe, hf_hub_download(repo_id='lime-j/GenSIRR', filename="GenSIRR.pt"))
# pipe.transformer.fuse_qkv_projections()
# pipe.transformer.set_attention_backend("_flash_3_hub")
pipe = pipe.to("cuda")
# optimize_pipeline_(pipe, image=Image.new("RGB", (512, 512)), prompt='prompt')

@spaces.GPU
def infer(input_image, seed=42, randomize_seed=False, steps=28, progress=gr.Progress(track_tqdm=True)):
    """
    Perform image editing using the FLUX.1 Kontext pipeline.
    
    This function takes an input image and a text prompt to generate a modified version
    of the image based on the provided instructions. It uses the FLUX.1 Kontext model
    for contextual image editing tasks.
    
    Args:
        input_image (PIL.Image.Image): The input image to be edited. Will be converted
            to RGB format if not already in that format.
        prompt (str): Text description of the desired edit to apply to the image.
            Examples: "Remove glasses", "Add a hat", "Change background to beach".
        seed (int, optional): Random seed for reproducible generation. Defaults to 42.
            Must be between 0 and MAX_SEED (2^31 - 1).
        randomize_seed (bool, optional): If True, generates a random seed instead of
            using the provided seed value. Defaults to False.
        guidance_scale (float, optional): Controls how closely the model follows the
            prompt. Higher values mean stronger adherence to the prompt but may reduce
            image quality. Range: 1.0-10.0. Defaults to 2.5.
        steps (int, optional): Controls how many steps to run the diffusion model for.
            Range: 1-30. Defaults to 28.
        progress (gr.Progress, optional): Gradio progress tracker for monitoring
            generation progress. Defaults to gr.Progress(track_tqdm=True).
    
    Returns:
        tuple: A 3-tuple containing:
            - PIL.Image.Image: The generated/edited image
            - int: The seed value used for generation (useful when randomize_seed=True)
            - gr.update: Gradio update object to make the reuse button visible
    
    Example:
        >>> edited_image, used_seed, button_update = infer(
        ...     input_image=my_image,
        ...     prompt="Add sunglasses",
        ...     seed=123,
        ...     randomize_seed=False,
        ...     guidance_scale=2.5
        ... )
    """
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    
    size = 512
    input_image = input_image.convert("RGB")
    if input_image.width < input_image.height:
        input_image = input_image.resize((size, int(size * input_image.height / input_image.width)))
    else:
        input_image = input_image.resize((int(size * input_image.width / input_image.height), size))
    tensor = TF.to_tensor(input_image)
    with torch.inference_mode():
        original_size = tensor.shape[-2:]
        padded_tensor, padding = pad_for_model(tensor, 16)
        batch_cpu = padded_tensor.unsqueeze(0)
        batch_device = batch_cpu.to('cuda')
        output = pipe(
            image=batch_device, 
            width = input_image.size[0],
            height = input_image.size[1],
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(seed),
        )
        #.images[0]
        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output

        output_tensor = output_tensor.squeeze(0).detach().cpu()
        h, w = original_size
        output_tensor = output_tensor[..., :h, :w]
        output_image = tensor_to_image(output_tensor)
    return output_image, seed, gr.Button(visible=True)

@spaces.GPU
def infer_example(input_image):
    image, seed, _ = infer(input_image)
    return image, seed

css="""
#col-container {
    margin: 0 auto;
    max-width: 960px;
}
"""

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# GenSIRR: Rectifying Latent Space for Generative SIRR 
        This is a demo for our generative single-image reflection removal model. 
        """)
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload the image for reflection removal", type="pil")
                with gr.Row():
                    run_button = gr.Button("Run")
                
                with gr.Accordion("Advanced Settings", open=False):
                    
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )
                    
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    
                    steps = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=30,
                        value=28,
                        step=1
                    )
                    
            with gr.Column():
                result = gr.Image(label="Result", show_label=False, interactive=False)
                reuse_button = gr.Button("Reuse this image", visible=False)
        
            
            
    gr.on(
        triggers=[run_button.click],
        fn = infer,
        inputs = [input_image, seed, randomize_seed, steps],
        outputs = [result, seed, reuse_button]
    )
    reuse_button.click(
        fn = lambda image: image,
        inputs = [result],
        outputs = [input_image]
    )

demo.launch(mcp_server=True)