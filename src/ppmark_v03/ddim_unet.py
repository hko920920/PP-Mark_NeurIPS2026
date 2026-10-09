"""
Optional UNet-based DDIM inversion scaffolding.

This module provides a thin wrapper around `diffusers` pipelines to allow
UNet-based inversion when weights are available. If diffusers/torch are not
installed, callers should fall back to the placeholder inversion in sync.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _is_sdxl_model(model_id: str) -> bool:
    tag = model_id.lower()
    return "sdxl" in tag or "stable-diffusion-xl" in tag


@dataclass(slots=True)
class DiffusersDDIMConfig:
    model_id: str
    device: str = "cuda"
    torch_dtype: str = "float32"
    num_steps: int = 50
    guidance_scale: float = 0.0
    negative_prompt: str = ""
    scheduler_kwargs: Optional[dict] = None
    prompt: str = ""
    scheduler: str = "ddim"
    attn_slicing: bool = False
    vae_slicing: bool = False
    grad_checkpointing: bool = False
    xformers: bool = False


class DiffusersDDIMInverter:
    def __init__(self, cfg: DiffusersDDIMConfig):
        try:
            import torch  # type: ignore
            from diffusers import (  # type: ignore
                DDIMInverseScheduler,
                DDIMScheduler,
                StableDiffusionPipeline,
                StableDiffusionXLPipeline,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("diffusers and torch are required for DDIM inversion") from exc

        dtype = getattr(torch, cfg.torch_dtype, torch.float16)
        use_sdxl = _is_sdxl_model(cfg.model_id)
        pipe_cls = StableDiffusionXLPipeline if use_sdxl else StableDiffusionPipeline
        kwargs = {"torch_dtype": dtype}
        if use_sdxl and cfg.torch_dtype in {"float16", "bfloat16"}:
            kwargs["variant"] = "fp16"
        try:
            pipe = pipe_cls.from_pretrained(cfg.model_id, **kwargs)
        except (TypeError, ValueError):
            kwargs.pop("variant", None)
            pipe = pipe_cls.from_pretrained(cfg.model_id, **kwargs)
        scheduler_kwargs = cfg.scheduler_kwargs or {}
        scheduler_name = cfg.scheduler.lower()
        if scheduler_name != "ddim":
            raise ValueError(f"Unsupported scheduler for inversion: {cfg.scheduler}")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, **scheduler_kwargs)
        self.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config, **scheduler_kwargs)
        pipe.to(cfg.device)
        if cfg.attn_slicing:
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        if cfg.vae_slicing:
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
        if cfg.grad_checkpointing:
            try:
                pipe.unet.enable_gradient_checkpointing()
            except Exception:
                pass
        if cfg.xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.cfg = cfg
        self.is_sdxl = use_sdxl
        self.prompt = cfg.prompt
        self.negative_prompt = cfg.negative_prompt

    def _invert_tensor(self, image: np.ndarray):
        """
        UNet-guided DDIM forward (encode image → latents → noisy latent at last timestep).
        Returns the full latent tensor (batch, channels, height, width).
        """
        import torch  # type: ignore

        pipe = self.pipe
        cfg = self.cfg
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        with torch.inference_mode():
            pil = pipe.image_processor.numpy_to_pil(image.astype(np.float32))
            vae_inputs = pipe.image_processor.preprocess(pil).to(device=pipe.device, dtype=pipe.dtype)
            latent_dist = pipe.vae.encode(vae_inputs).latent_dist
            latents = latent_dist.mode()
            latents = latents * pipe.vae.config.scaling_factor

            do_cfg = float(cfg.guidance_scale) > 0.0
            if self.is_sdxl:
                enc_res = pipe.encode_prompt(
                    self.prompt or "",
                    device=pipe.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=do_cfg,
                    negative_prompt=self.negative_prompt if do_cfg else None,
                )
                prompt_embeds = None
                pooled_embeds = None
                negative_prompt_embeds = None
                negative_pooled_embeds = None
                if isinstance(enc_res, tuple):
                    if do_cfg and len(enc_res) >= 4:
                        prompt_embeds = enc_res[0]
                        negative_prompt_embeds = enc_res[1]
                        pooled_embeds = enc_res[2]
                        negative_pooled_embeds = enc_res[3]
                        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                        pooled_embeds = (
                            torch.cat([negative_pooled_embeds, pooled_embeds], dim=0)
                            if negative_pooled_embeds is not None
                            else pooled_embeds
                        )
                    elif len(enc_res) >= 3:
                        prompt_embeds = enc_res[0]
                        pooled_embeds = enc_res[2]
                    else:
                        prompt_embeds, pooled_embeds = enc_res
                else:
                    prompt_embeds, pooled_embeds = enc_res, None
                b, c, h, w = latents.shape
                proj_dim = getattr(pipe.text_encoder_2.config, "projection_dim", prompt_embeds.shape[-1])
                add_time_ids = pipe._get_add_time_ids(
                    (w * 8, h * 8),  # original size
                    (0, 0),  # crop coords
                    (w * 8, h * 8),  # target size
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=proj_dim,
                ).to(pipe.device)
                add_time_ids = add_time_ids.repeat(prompt_embeds.shape[0], 1)
            else:
                if hasattr(pipe, "_encode_prompt"):
                    prompt_embeds = pipe._encode_prompt(  # type: ignore[attr-defined]
                        self.prompt or "",
                        device=pipe.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=do_cfg,
                        negative_prompt=self.negative_prompt if do_cfg else None,
                    )
                else:
                    enc_res = pipe.encode_prompt(
                        self.prompt or "",
                        device=pipe.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=do_cfg,
                        negative_prompt=self.negative_prompt if do_cfg else None,
                    )
                    if isinstance(enc_res, tuple):
                        prompt_embeds = enc_res[0]
                        negative_prompt_embeds = enc_res[1] if len(enc_res) > 1 else None
                    else:
                        prompt_embeds = enc_res
                        negative_prompt_embeds = None
                    if do_cfg:
                        if negative_prompt_embeds is None:
                            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                pooled_embeds = None
                add_time_ids = None

            self.inverse_scheduler.set_timesteps(cfg.num_steps, device=pipe.device)
            x_t = latents
            pooled_local = pooled_embeds
            for t in self.inverse_scheduler.timesteps:
                if self.is_sdxl:
                    proj_dim = getattr(pipe.text_encoder_2.config, "projection_dim", prompt_embeds.shape[-1])
                    if pooled_local is None:
                        pooled_local = torch.zeros(
                            (prompt_embeds.shape[0], proj_dim), device=pipe.device, dtype=prompt_embeds.dtype
                        )
                    cond_kwargs = {"text_embeds": pooled_local, "time_ids": add_time_ids}
                else:
                    cond_kwargs = None
                latent_input = torch.cat([x_t] * 2) if do_cfg else x_t
                latent_input = self.inverse_scheduler.scale_model_input(latent_input, t)
                if cond_kwargs is None:
                    noise_pred = pipe.unet(
                        latent_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]
                else:
                    noise_pred = pipe.unet(
                        latent_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=cond_kwargs,
                        return_dict=False,
                    )[0]
                if do_cfg:
                    noise_uncond, noise_text = noise_pred.chunk(2)
                    noise_pred = noise_uncond + cfg.guidance_scale * (noise_text - noise_uncond)
                step = self.inverse_scheduler.step(noise_pred, t, x_t)
                x_t = step.prev_sample

            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)

        return x_t

    def invert_latents(self, image: np.ndarray) -> np.ndarray:
        """Return full latent tensor (channels, height, width) as numpy."""
        import torch  # type: ignore
        x_t = self._invert_tensor(image)
        return x_t[0].detach().to(dtype=torch.float32).cpu().numpy()

    def invert(self, image: np.ndarray) -> np.ndarray:
        """Return channel-0 latent (height, width) for watermark detection."""
        return self.invert_latents(image)[0]

    def invert_latents_torch(self, image):
        """Return full latent tensor (batch, channels, height, width) with gradients enabled."""
        import torch  # type: ignore

        pipe = self.pipe
        cfg = self.cfg

        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.shape[1] != 3 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)

        image = image.to(device=pipe.device, dtype=pipe.dtype)
        vae_inputs = image * 2.0 - 1.0
        latent_dist = pipe.vae.encode(vae_inputs).latent_dist
        latents = latent_dist.mode() * pipe.vae.config.scaling_factor

        do_cfg = float(cfg.guidance_scale) > 0.0
        with torch.no_grad():
            if self.is_sdxl:
                enc_res = pipe.encode_prompt(
                    self.prompt or "",
                    device=pipe.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=do_cfg,
                    negative_prompt=self.negative_prompt if do_cfg else None,
                )
                prompt_embeds = None
                pooled_embeds = None
                negative_prompt_embeds = None
                negative_pooled_embeds = None
                if isinstance(enc_res, tuple):
                    if do_cfg and len(enc_res) >= 4:
                        prompt_embeds = enc_res[0]
                        negative_prompt_embeds = enc_res[1]
                        pooled_embeds = enc_res[2]
                        negative_pooled_embeds = enc_res[3]
                        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                        pooled_embeds = (
                            torch.cat([negative_pooled_embeds, pooled_embeds], dim=0)
                            if negative_pooled_embeds is not None
                            else pooled_embeds
                        )
                    elif len(enc_res) >= 3:
                        prompt_embeds = enc_res[0]
                        pooled_embeds = enc_res[2]
                    else:
                        prompt_embeds, pooled_embeds = enc_res
                else:
                    prompt_embeds, pooled_embeds = enc_res, None
                b, c, h, w = latents.shape
                proj_dim = getattr(pipe.text_encoder_2.config, "projection_dim", prompt_embeds.shape[-1])
                add_time_ids = pipe._get_add_time_ids(
                    (w * 8, h * 8),
                    (0, 0),
                    (w * 8, h * 8),
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=proj_dim,
                ).to(pipe.device)
                add_time_ids = add_time_ids.repeat(prompt_embeds.shape[0], 1)
            else:
                if hasattr(pipe, "_encode_prompt"):
                    prompt_embeds = pipe._encode_prompt(  # type: ignore[attr-defined]
                        self.prompt or "",
                        device=pipe.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=do_cfg,
                        negative_prompt=self.negative_prompt if do_cfg else None,
                    )
                else:
                    enc_res = pipe.encode_prompt(
                        self.prompt or "",
                        device=pipe.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=do_cfg,
                        negative_prompt=self.negative_prompt if do_cfg else None,
                    )
                    if isinstance(enc_res, tuple):
                        prompt_embeds = enc_res[0]
                        negative_prompt_embeds = enc_res[1] if len(enc_res) > 1 else None
                    else:
                        prompt_embeds = enc_res
                        negative_prompt_embeds = None
                    if do_cfg:
                        if negative_prompt_embeds is None:
                            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                pooled_embeds = None
                add_time_ids = None

        self.inverse_scheduler.set_timesteps(cfg.num_steps, device=pipe.device)
        x_t = latents
        pooled_local = pooled_embeds
        # Gradient checkpointing is only active in train mode, so temporarily flip the UNet if requested.
        unet_was_training = pipe.unet.training
        if cfg.grad_checkpointing and not unet_was_training:
            pipe.unet.train()
        for t in self.inverse_scheduler.timesteps:
            if self.is_sdxl:
                proj_dim = getattr(pipe.text_encoder_2.config, "projection_dim", prompt_embeds.shape[-1])
                if pooled_local is None:
                    pooled_local = torch.zeros(
                        (prompt_embeds.shape[0], proj_dim), device=pipe.device, dtype=prompt_embeds.dtype
                    )
                cond_kwargs = {"text_embeds": pooled_local, "time_ids": add_time_ids}
            else:
                cond_kwargs = None
            latent_input = torch.cat([x_t] * 2) if do_cfg else x_t
            latent_input = self.inverse_scheduler.scale_model_input(latent_input, t)
            if cond_kwargs is None:
                noise_pred = pipe.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
            else:
                noise_pred = pipe.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=cond_kwargs,
                    return_dict=False,
                )[0]
            if do_cfg:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + cfg.guidance_scale * (noise_text - noise_uncond)
            step = self.inverse_scheduler.step(noise_pred, t, x_t)
            x_t = step.prev_sample
        if cfg.grad_checkpointing and not unet_was_training:
            pipe.unet.eval()

        x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)
        return x_t
