# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import math
import numpy as np
import torch

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from ..models.unet import RoviModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class SaveOutputHook:
    def __init__(
            self,
            height,
            width,
        ):
        self.outputs = []
        self.height = height
        self.width = width
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []

    def get_attn_map(self):
        last_layter_output = self.outputs[-1].cpu().detach().numpy()
        length = last_layter_output.shape[1]
        height = math.sqrt(length * self.height / self.width)
        last_layter_output = rearrange(last_layter_output, "f (h w) c -> 1 c f h w", h=int(height))
        last_layter_output = last_layter_output.mean(axis=1, keepdims=True)
        for t in range(last_layter_output.shape[2]):
            _range = np.max(last_layter_output[:,:,t,:,:]) - np.min(last_layter_output[:,:,t,:,:])
            last_layter_output[:,:,t,:,:] = (last_layter_output[:,:,t,:,:] - np.min(last_layter_output[:,:,t,:,:])) / _range
        last_layter_output = torch.Tensor(last_layter_output)
        return last_layter_output


@dataclass
class RoviPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]
    attn_map: torch.Tensor
    latent: torch.Tensor
    t: int
    mask: torch.Tensor


class RoviPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: RoviModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            # pix2pix has two  negative embeddings, and unlike in other pipelines latents are ordered [prompt_embeds, negative_prompt_embeds, negative_prompt_embeds]
            prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds, negative_prompt_embeds])

        return prompt_embeds

    @torch.no_grad()
    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = self.vae.decode(latents).sample
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_image_latents(
        self, image, batch_size, num_images_per_prompt, device, do_classifier_free_guidance, generator=None
    ):
        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            image_latents = image
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                image_latents = [self.vae.encode(image[i : i + 1]).latent_dist.mode() for i in range(batch_size)]
                image_latents = torch.cat(image_latents, dim=0)
            else:
                image_latents = self.vae.encode(image).latent_dist.mode()

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand image_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        if do_classifier_free_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

        
    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        img_condition: torch.FloatTensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        return_attn_map: bool = False,
        attn_map_t: int = 0,
        visual_timestep: int = None,
        mid_sample_timestep: int = None,
        **kwargs,
    ):
        # Default height and width to unet
        height = img_condition.shape[3]
        width = img_condition.shape[4]
        video_length = img_condition.shape[2]

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0 and image_guidance_scale >= 1.0

        # Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        visual_timestep = visual_timestep if visual_timestep is not None else np.random.randint(0, len(timesteps))
        if return_attn_map:
            print(f"visual timestep: {visual_timestep}")
        if mid_sample_timestep is not None:
            print(f"mid sample timestep: {mid_sample_timestep}")

        # Prepare Image latents
        image_latents = self.prepare_image_latents(
            img_condition,
            batch_size,
            num_videos_per_prompt,
            device,
            do_classifier_free_guidance,
            generator,
        )

        # Prepare latent variables
        # num_channels_latents = self.unet.in_channels
        num_channels_latents = image_latents.shape[1]
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        save_output = SaveOutputHook(height=height, width=width)
        # save the attention map
        if return_attn_map:
            attn_name = 'up_blocks.3.attentions.2.transformer_blocks.0.attn2'
            # attn_name = 'up_blocks.3.attentions.2.transformer_blocks.0.attn1'
            for name, module in self.unet.named_modules():
                if attn_name in name:
                    hook_handle = module.register_forward_hook(save_output)
            mask = self.unet(latents, attn_map_t - 1000 // num_inference_steps, encoder_hidden_states=text_embeddings[0:1], img_condition=image_latents[0:1]).mask.to(dtype=latents_dtype)
            attn_map = save_output.get_attn_map()
            save_output.clear()
            hook_handle.remove()
            print("attention map saved")
            return RoviPipelineOutput(videos=None, attn_map=attn_map, latent=None, t=None, mask=mask)
        
        print(f"visual timestep {visual_timestep}")
        output_latent = None
        output_t = None
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # split the three forward to avoid out of memory
                noise_preds = []
                for j, (latent, text_embedding, image_latent) in enumerate(zip(latent_model_input, text_embeddings, image_latents)):
                    # predict the noise residual
                    noise_pred = self.unet(latent.unsqueeze(0), t, encoder_hidden_states=text_embedding.unsqueeze(0), img_condition=image_latent.unsqueeze(0)).sample.to(dtype=latents_dtype)
                    noise_preds.append(noise_pred)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_preds
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_image)
                        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                if not return_attn_map and i == visual_timestep:
                    output_latent = latents
                    output_t = t
                    print(f"latents in timestep {t} saved")

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

                if mid_sample_timestep is not None and mid_sample_timestep == i:
                    break

        # Post-processing
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return RoviPipelineOutput(videos=video, attn_map=None, latent=output_latent, t=output_t, mask=None)