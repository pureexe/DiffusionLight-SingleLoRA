from diffusers import StableDiffusionXLPipeline
import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
import gc

class MyUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            # SDXL - style
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config.addition_embed_type == "image_hint":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb, hint = self.add_embedding(image_embs, hint)
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            # Kadinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = mid_block_additional_residual is None and down_block_additional_residuals is not None

        down_block_res_samples = (sample,)
        down_ft = {}
        for i, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_block_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_block_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, scale=lora_scale)

                if is_adapter and len(down_block_additional_residuals) > 0:
                    sample += down_block_additional_residuals.pop(0)

            down_block_res_samples += res_samples

            down_ft[i] = sample.detach()

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        mid_ft = {}
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_block_additional_residuals) > 0
                and sample.shape == down_block_additional_residuals[0].shape
            ):
                sample += down_block_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        mid_ft[0] = sample.detach()

        # 5. up
        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):

            # if i > np.max(up_ft_indices):
            #     break

            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    scale=lora_scale,
                )

            up_ft[i] = sample.detach()#.cpu()
            # if i in [0, 1, 2]:
            #     up_ft[i] = sample.detach().cpu()

        output = {}
        output['down_ft'] = down_ft
        output['mid_ft'] = mid_ft
        output['up_ft'] = up_ft
        return output

class OneStepSDXLPipeline(StableDiffusionXLPipeline):
    @torch.no_grad()
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        original_size: Optional[Tuple[int, int]] = None,
        target_size: Optional[Tuple[int, int]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        height=1024,
        width=1024,
    ):
        
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = randn_tensor(
            latents.shape, generator=generator,
            device=latents.device, dtype=latents.dtype, layout=latents.layout
        )
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        add_text_embeds = pooled_prompt_embeds
        
        # try:
        #     # print("HERRRRRRRRRRRRR")
        #     add_time_ids = self._get_add_time_ids(
        #         original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        #     )
        #     self.add_time_ids = add_time_ids
        #     # print("XXXXXXXXXXXXXXXXXXXXx")
        # except:
        #     add_time_ids = self.add_time_ids
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        self.add_time_ids = add_time_ids

        add_text_embeds = add_text_embeds.to(latents.device)
        add_time_ids = add_time_ids.to(latents.device).repeat(1, 1)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        
        unet_output = self.unet(
            latents_noisy,
            t,
            up_ft_indices=up_ft_indices,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
        )
        return unet_output


class SDXLFeaturizer:
    def __init__(self, pipeline, device="cuda:0"):
        self.pipe = pipeline
        self.device = device

        self.pipe.vae.decoder = None
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.config)
        self.pipe = self.pipe.to(self.device)
        # onestep_pipe.enable_attention_slicing()
        self.pipe.enable_xformers_memory_efficient_attention()

        self.prompt_embeds = None
        self.pooled_prompt_embeds = None

    @torch.no_grad()
    def forward(self, 
                img_tensor,
                prompt, 
                t=261, 
                up_ft_index=1, 
                ensemble_size=8,
                text_encoder_lora_scale=None,
                cross_attention_kwargs = None,
                height=1024,
                width=1024,
                generator=None,
                use_cached_prompt=True
        ):
        '''
        Args:
            img_tensor: should be a single torch tensor in the shape of [1, C, H, W] or [C, H, W]
            prompt: the prompt to use, a string
            t: the time step to use, should be an int in the range of [0, 1000]
            up_ft_index: which upsampling block of the U-Net to extract feature, you can choose [0, 1, 2, 3]
            ensemble_size: the number of repeated images used in the batch to extract features
        Return:
            unet_ft: a torch tensor in the shape of [1, c, h, w]
        '''
        img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).to("cuda") # ensem, c, h, w
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        if (self.prompt_embeds is None) or (not use_cached_prompt):
            (
                prompt_embeds,
                _,
                pooled_prompt_embeds,
                _,
            ) = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt, # we use the same prompt for both encoders
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False, # [1, 77, dim]
                negative_prompt=None,
                negative_prompt_2=None,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=None,
                negative_pooled_prompt_embeds=None,
                lora_scale=text_encoder_lora_scale,
            )
            # print("HERE")
            self.prompt_embeds = prompt_embeds
            self.pooled_prompt_embeds = pooled_prompt_embeds
        else:
            prompt_embeds = self.prompt_embeds
            pooled_prompt_embeds = self.pooled_prompt_embeds

        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1)
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=torch.float16)

        generator = generator if generator is not None else torch.Generator().manual_seed(0)
        unet_ft_all = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            generator=generator,
            cross_attention_kwargs=cross_attention_kwargs,
            height=height,
            width=width
        )
        if up_ft_index is not None:
            raise NotImplementedError
            unet_ft = unet_ft_all['up_ft'][up_ft_index] # ensem, c, h, w
            unet_ft = unet_ft.mean(0, keepdim=True) # 1,c,h,w
        else:
            unet_down_ft = [
                unet_ft_all['down_ft'][i].mean(0, keepdim=True) for i in range(len(unet_ft_all['down_ft']))
            ]
            unet_mid_ft = [
                unet_ft_all['mid_ft'][i].mean(0, keepdim=True) for i in range(len(unet_ft_all['mid_ft']))
            ]
            unet_up_ft = [
                unet_ft_all['up_ft'][i].mean(0, keepdim=True) for i in range(len(unet_ft_all['up_ft']))
            ]
        return unet_down_ft, unet_mid_ft, unet_up_ft

# class SDXLFeaturizerTraining:
#     def __init__(self, pipeline, device="cuda:0"):
#         self.pipe = pipeline
#         self.device = device

#         self.pipe.vae.decoder = None
#         self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.config)
#         self.pipe = self.pipe.to(self.device)
#         # onestep_pipe.enable_attention_slicing()
#         self.pipe.enable_xformers_memory_efficient_attention()
#         self.prompt_embeds = None
#         self.pooled_prompt_embeds = None
#         gc.collect()

#     @torch.no_grad()
#     def forward(self, 
#                 img_tensor,
#                 prompt, 
#                 t=261, 
#                 up_ft_index=1, 
#                 ensemble_size=8,
#                 text_encoder_lora_scale=None,
#                 cross_attention_kwargs = None,
#                 height=1024,
#                 width=1024,
#                 generator=None,
#                 use_cached_prompt=True,
#         ):
#         '''
#         Args:
#             img_tensor: should be a single torch tensor in the shape of [1, C, H, W] or [C, H, W]
#             prompt: the prompt to use, a string
#             t: the time step to use, should be an int in the range of [0, 1000]
#             up_ft_index: which upsampling block of the U-Net to extract feature, you can choose [0, 1, 2, 3]
#             ensemble_size: the number of repeated images used in the batch to extract features
#         Return:
#             unet_ft: a torch tensor in the shape of [1, c, h, w]
#         '''
#         # img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).to("cuda") # ensem, c, h, w
#         if (self.prompt_embeds is None) or (not use_cached_prompt):
#             (
#                 prompt_embeds,
#                 _,
#                 pooled_prompt_embeds,
#                 _,
#             ) = self.pipe.encode_prompt(
#                 prompt=prompt,
#                 prompt_2=prompt, # we use the same prompt for both encoders
#                 device=self.device,
#                 num_images_per_prompt=1,
#                 do_classifier_free_guidance=False, # [1, 77, dim]
#                 negative_prompt=None,
#                 negative_prompt_2=None,
#                 prompt_embeds=None,
#                 negative_prompt_embeds=None,
#                 pooled_prompt_embeds=None,
#                 negative_pooled_prompt_embeds=None,
#                 lora_scale=text_encoder_lora_scale,
#             )
#             self.prompt_embeds = prompt_embeds
#             self.pooled_prompt_embeds = pooled_prompt_embeds
#         else:
#             prompt_embeds = self.prompt_embeds
#             pooled_prompt_embeds = self.pooled_prompt_embeds

#         prompt_embeds = prompt_embeds.repeat(img_tensor.shape[0], 1, 1)
#         prompt_embeds = prompt_embeds.to(device=self.device, dtype=torch.float16)

#         generator = generator if generator is not None else torch.Generator().manual_seed(0)
#         unet_ft_all = self.pipe(
#             img_tensor=img_tensor,
#             t=t,
#             up_ft_indices=[up_ft_index],
#             prompt_embeds=prompt_embeds,
#             pooled_prompt_embeds=pooled_prompt_embeds,
#             generator=generator,
#             cross_attention_kwargs=cross_attention_kwargs,
#             height=height,
#             width=width
#         )

#         if use_cached_prompt:
#             self.pipe.text_encoder = None
#             self.pipe.text_encoder_2 = None
#             self.pipe.tokenizer = None
#             self.pipe.tokenizer_2 = None
#             gc.collect()

#         return [unet_ft_all['up_ft'][i] for i in range(len(unet_ft_all['up_ft']))]
#         # if up_ft_index is not None:
#         #     unet_ft = unet_ft_all['up_ft'][up_ft_index] # ensem, c, h, w
#         #     unet_ft = unet_ft.mean(0, keepdim=True) # 1,c,h,w
#         # else:
#         #     unet_ft = [
#         #         unet_ft_all['up_ft'][i].mean(0, keepdim=True) for i in range(len(unet_ft_all['up_ft']))
#         #     ]
#         # return unet_ft