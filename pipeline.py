import os
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from typing import Dict, Any, Optional, List, Callable, Union
import torch
import torch.nn as nn
import numpy as np
from diffusers import FluxKontextPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from peft import LoraConfig, LoraModel, get_peft_model

torch.set_float32_matmul_precision('medium')
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
    (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
    (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720),
    (1504, 688), (1568, 672),
]


def _resolve_vae_path(user_path: Optional[str] = None) -> str:
    """Resolve where to load the VAE weights from."""
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        user_path,
        os.environ.get("FLUX_VAE_PATH"),
        os.environ.get("VAE_PATH"),
        repo_root / "vae_merged",
        "/home/s1023244038/XReflection/vae_merged",
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists():
            return str(candidate_path)

    raise FileNotFoundError(
        "Could not locate VAE weights. Please set the `FLUX_VAE_PATH` "
        "environment variable to the directory that contains the merged VAE."
    )

def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

def calculate_shift(image_seq_len, base_image_seq_len, max_image_seq_len, base_shift, max_shift):
    return base_shift + (max_shift - base_shift) * (image_seq_len - base_image_seq_len) / (
        max_image_seq_len - base_image_seq_len
    )

def retrieve_timesteps(
    scheduler, num_inference_steps: Optional[int] = None, device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None, sigmas: Optional[List[float]] = None, **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

    if timesteps is not None:
        scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device, timesteps=timesteps, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif num_inference_steps is not None:
        scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        raise ValueError("Either `num_inference_steps` or `timesteps` or `sigmas` has to be passed.")
    return timesteps, num_inference_steps



class GenSIRR(nn.Module):
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(self, model_path, train_dit: bool = True, vae_path: Optional[str] = None):
        super().__init__()
        self.train_dit = train_dit
        pipe = FluxKontextPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16, )
        self.dtype = torch.bfloat16
        self.vae = pipe.vae

        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.text_encoder_2 = None #pipe.text_encoder_2
        self.tokenizer_2 = None #pipe.tokenizer_2
        self.transformer = pipe.transformer
        self.scheduler = pipe.scheduler
        # self.image_encoder = pipe.image_encoder.to("cuda")
        self.image_processor = pipe.image_processor
        
        self.latent_channels = self.transformer.config.in_channels // 4
        self.vae_scale_factor = pipe.vae_scale_factor
        self.joint_attention_kwargs = getattr(pipe, '_joint_attention_kwargs', None)
        self._execution_device = pipe._execution_device
        self.default_sample_size = pipe.default_sample_size
        self.interrupt = False
        self.tokenizer_max_length = pipe.tokenizer_max_length
        self.transformer.enable_gradient_checkpointing()
        self.cached_prompt_embeds = torch.nn.Parameter(torch.load("prompt_embeds.pth", map_location='cpu'))
        self.cached_pooled_prompt_embeds = torch.nn.Parameter(torch.load("pooled_prompt_embeds.pth", map_location='cpu'))
        self.cached_text_ids = torch.nn.Parameter(torch.load("text_ids.pth", map_location='cpu'))
        del pipe
    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._prepare_latent_image_ids
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._unpack_latents
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def progress_bar(self, iterable):
        return iterable

    def maybe_free_model_hooks(self):
        pass


    def check_inputs(
        self, prompt, prompt_2, height, width, negative_prompt=None, negative_prompt_2=None,
        prompt_embeds=None, negative_prompt_embeds=None, pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None, callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        if callback_on_step_end_tensor_inputs is not None and not all(k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs):
            raise ValueError(f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}")
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        if prompt_2 is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt_2` and `prompt_embeds`.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if prompt_2 is not None and not isinstance(prompt_2, (str, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot forward both `negative_prompt` and `negative_prompt_embeds`.")
        if negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot forward both `negative_prompt_2` and `negative_prompt_embeds`.")
        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError("If `prompt_embeds` are provided, `pooled_prompt_embeds` must also be passed.")
        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError("If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` must also be passed.")

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = self.text_encoder_2.device if self.text_encoder_2 is not None else self.text_encoder.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)


        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_clip_prompt_embeds
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = self.text_encoder.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        # if isinstance(self, TextualInversionLoaderMixin):
        #     prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(self.text_encoder.device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = self.text_encoder.device
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None:
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )


        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)


        
        return prompt_embeds, pooled_prompt_embeds, text_ids


    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents
    def prepare_latents(self, image, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)

        image_latents = image_ids = None
        if image is not None:
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != self.latent_channels:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents = image
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                image_latents = torch.cat([image_latents], dim=0)

            image_latent_height, image_latent_width = image_latents.shape[2:]
            image_latents = self._pack_latents(
                image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
            )
            image_ids = self._prepare_latent_image_ids(
                batch_size, image_latent_height // 2, image_latent_width // 2, device, dtype
            )
            # image ids are the same as latent ids with the first dimension set to 1 instead of 0
            image_ids[..., 0] = 1

        latent_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents, latent_ids, image_ids 

    def forward(
        self, image: PipelineImageInput = None, prompt: Optional[str] = None, prompt_2: Optional[str] = None,
        negative_prompt: Optional[str] = None, negative_prompt_2: Optional[str] = None,
        height: Optional[int] = None, width: Optional[int] = None, num_inference_steps: int = 28,
        guidance_scale: float = 3.5, num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None, output_type: Optional[str] = "pil",
        return_dict: bool = True, max_area: int = 1024**2, _auto_resize: bool = True,
        **kwargs
    ):
        joint_attention_kwargs = kwargs.get("joint_attention_kwargs")
        prompt_embeds = kwargs.get("prompt_embeds")
        pooled_prompt_embeds = kwargs.get("pooled_prompt_embeds")
        negative_prompt_embeds = kwargs.get("negative_prompt_embeds")
        negative_pooled_prompt_embeds = kwargs.get("negative_pooled_prompt_embeds")
        ip_adapter_image = kwargs.get("ip_adapter_image")
        ip_adapter_image_embeds = kwargs.get("ip_adapter_image_embeds")
        negative_ip_adapter_image = kwargs.get("negative_ip_adapter_image")
        negative_ip_adapter_image_embeds = kwargs.get("negative_ip_adapter_image_embeds")
        callback_on_step_end = kwargs.get("callback_on_step_end")
        callback_on_step_end_tensor_inputs = kwargs.get("callback_on_step_end_tensor_inputs", ["latents"])
        max_sequence_length = kwargs.get("max_sequence_length", 512)

        sigmas = kwargs.get("sigmas")
        height, width = image.shape[2], image.shape[3]
        # height = height or self.default_sample_size * self.vae_scale_factor
        # width = width or self.default_sample_size * self.vae_scale_factor
        original_height, original_width = height, width
        aspect_ratio = width / height
        # width = round((max_area * aspect_ratio) ** 0.5)
        # height = round((max_area / aspect_ratio) ** 0.5)
        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of
        if height != original_height or width != original_width:
            logger.warning(f"Resizing to {height}x{width} to fit model requirements.")

        prompt = 'please remove the reflection in this image'
        prompt_2 = 'please remove the reflection in this image'

        if self.text_encoder_2 is not None:
            self.text_encoder.max_position_embeddings = 77
            self.text_encoder_2.max_position_embeddings = 512
        self.check_inputs(
            prompt, prompt_2, height, width, negative_prompt, negative_prompt_2,
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
            negative_pooled_prompt_embeds, callback_on_step_end_tensor_inputs
        )
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str): batch_size = 1
        elif prompt is not None and isinstance(prompt, list): batch_size = len(prompt)
        else: batch_size = prompt_embeds.shape[0]
        device = self.text_encoder.device
        lora_scale = self.joint_attention_kwargs.get("scale") if self.joint_attention_kwargs is not None else None
        
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds, pooled_prompt_embeds, text_ids = self.cached_prompt_embeds, self.cached_pooled_prompt_embeds, self.cached_text_ids
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            img = image[0] if isinstance(image, list) else image
            image_height, image_width = image.shape[2], image.shape[3]
            image = self.image_processor.resize(image, image_height, image_width)
            image = self.image_processor.preprocess(image, image_height, image_width)

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents, latent_ids, image_ids = self.prepare_latents(
            image, batch_size * num_images_per_prompt, num_channels_latents, height, width,
            prompt_embeds.dtype, device, generator, latents
        )
        if image_ids is not None:
            latent_ids = torch.cat([latent_ids, image_ids], dim=0)

        mu = calculate_shift(latents.shape[1], self.scheduler.config.get("base_image_seq_len", 256),
                             self.scheduler.config.get("max_image_seq_len", 4096),
                             self.scheduler.config.get("base_shift", 0.5),
                             self.scheduler.config.get("max_shift", 1.15))
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
        
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)


        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {} 
        self.scheduler.set_begin_index(0)
        for i, t in self.progress_bar(enumerate(timesteps)):
            if self.interrupt: break

            self._current_timestep = t


            latent_model_input = latents
            
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1)
            
            

            noise_pred = self.transformer(
                hidden_states=latent_model_input, 
                timestep=t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype) / 1000,
                guidance=guidance, 
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds, 
                txt_ids=text_ids, 
                img_ids=latent_ids,
                joint_attention_kwargs=self.joint_attention_kwargs, return_dict=False
            )[0][:, :latents.size(1)]

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)



        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            # image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict: return (image,)
        # self.output = image
        return (image + 1) / 2
    
    # @staticmethod
    def encode_image(self, images: torch.Tensor):
        """
        Encodes the images into tokens and ids for FLUX pipeline.
        """
        images = self.image_processor.preprocess(images)
        images = images.to(self.text_encoder.device).to(self.dtype)
        images = self.vae.encode(images).latent_dist.sample()
        images = (
            images - self.vae.config.shift_factor
        ) * self.vae.config.scaling_factor
        images_tokens = self._pack_latents(images, *images.shape)
        images_ids = self._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2],
            images.shape[3],
            self.text_encoder.device,
            self.dtype,
        )
        if images_tokens.shape[1] != images_ids.shape[0]:
            images_ids = self._prepare_latent_image_ids(
                images.shape[0],
                images.shape[2] // 2,
                images.shape[3] // 2,
                self.text_encoder.device,
                self.dtype,
            )
        return images_tokens, images_ids

if __name__ == "__main__":
    with torch.no_grad():
        from PIL import Image
        opt = {
            "model": "/home/s1023244038/kontext/",
        }
        model = FluxModel(opt)
        
        image = Image.open("/home/s1023244038/sirs/test/Nature/blended/1_143.jpg")
        prompt = ""
        prompt_2 = ""
        out = model(image=image, prompt=prompt, prompt_2=prompt_2)
        
        out[0].save("output.png")
