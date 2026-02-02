import torch
from tqdm.auto import tqdm
from typing import Any, Callable, Dict, List, Optional, Union
from torch.optim.adam import Adam
import torch.nn.functional as nnf
import numpy as np
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 3.5

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

## Inversion
@torch.no_grad()
def invert(
    start_latents,
    prompt,
    pipe,
    guidance_scale=3.5,
    num_inference_steps=80,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    max_sequence_length: int = 256,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
):

    # Encode prompt
    # text_embeddings = pipe._encode_prompt(
    #     prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    # )
    lora_scale = (
            joint_attention_kwargs.get("scale", None) if joint_attention_kwargs is not None else None
        )
    (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

    if do_classifier_free_guidance:
        # if skip_guidance_layers is not None:
        #     original_prompt_embeds = prompt_embeds
        #     original_pooled_prompt_embeds = pooled_prompt_embeds
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
   
    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)
    timesteps = timesteps.add(1)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        # noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        timestep = t.expand(latent_model_input.shape[0])
        noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]


        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        #反向更新步骤(重新安排更新步骤，使x(t)（新潜在数）成为x（t-1)（当前潜在数）的函数）
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt() / alpha_t.sqrt()) + (
            1 - alpha_t_next
        ).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents)


# Sample function (regular DDIM)
@torch.no_grad()
def sample(
    prompt,
    pipe,
    start_step=0,
    start_latents=None,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    max_sequence_length: int = 256,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None
):

    # Encode prompt
    # text_embeddings = pipe._encode_prompt(
    #     prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    # )
    lora_scale = (
            joint_attention_kwargs.get("scale", None) if joint_attention_kwargs is not None else None
        )
    (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
    
    if do_classifier_free_guidance:
        # if skip_guidance_layers is not None:
        #     original_prompt_embeds = prompt_embeds
        #     original_pooled_prompt_embeds = pooled_prompt_embeds
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        pooled_prompt_embeds = uncond_embeddings

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps)):

        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        # noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        timestep = t.expand(latent_model_input.shape[0])
        noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Normally we'd rely on the scheduler to handle the update step:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Instead, let's do it ourselves:
        prev_t = max(1, t.item() - (1000 // num_inference_steps))  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt

    # Post-processing
    # images = pipe.decode_latents(latents)
    # images = pipe.numpy_to_pil(images)

    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

    images = pipe.vae.decode(latents, return_dict=False)[0]
    images = pipe.image_processor.postprocess(images, output_type="pil")
    # images = pipe.numpy_to_pil(images)

    return images

# ---------------------------------
# 图像反演起始
def prev_step(pipe, noise_pred: Union[torch.FloatTensor, np.ndarray], t: int, latent_cur: Union[torch.FloatTensor, np.ndarray]):
        prev_t = max(1, t.item() - (1000 // NUM_DDIM_STEPS))  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latent_cur - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1 - alpha_t_prev).sqrt() * noise_pred
        prev_sample = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt
        # prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        # alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        # alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        # beta_prod_t = 1 - alpha_prod_t
        # pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        # pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        # prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

# uncond_embeddings, cond_embeddings其中一个为空
def get_noise_pred_single(pipe, latents, t, uncond_embeddings, cond_embeddings, joint_attention_kwargs):
        latent_model_input = latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        timestep = t.expand(latent_model_input.shape[0])
        noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=uncond_embeddings,
                    pooled_projections=cond_embeddings,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]
        return noise_pred

def get_noise_pred(pipe, latent_cur, t, uncond_embeddings, cond_embeddings, joint_attention_kwargs):
        latent_model_input = torch.cat([latent_cur] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        timestep = t.expand(latent_model_input.shape[0])
        guidance_scale = GUIDANCE_SCALE

        noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=uncond_embeddings,
                    pooled_projections=cond_embeddings,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = prev_step(noise_pred, t, latent_cur)
        return latents

# null optimization
@torch.no_grad()
def null_optimization(
    ddim_latents, 
    num_inner_steps, 
    early_stop_epsilon,
    prompt,
    pipe,
    guidance_scale=3.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device=device,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    max_sequence_length: int = 256,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
):
    lora_scale = (
            joint_attention_kwargs.get("scale", None) if joint_attention_kwargs is not None else None
        )
    (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
    
    if do_classifier_free_guidance:
        cond_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        uncond_embeddings = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    uncond_embeddings_list = []
    latent_cur = ddim_latents[-1]
    for i in range(num_inference_steps):
        uncond_embeddings = uncond_embeddings.clone().detach()
        uncond_embeddings.requires_grad = True
        # 单独优化每个uncond_embeddings=(1,77,768)
        optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
        latent_prev = ddim_latents[len(ddim_latents) - i - 2]
        t = pipe.scheduler.timesteps[i]
        # 条件部分不进行优化
        with torch.no_grad():
            noise_pred_cond = get_noise_pred_single(pipe=pipe, latents=latent_cur, t=t, uncond_embeddings=None, cond_embeddings=cond_embeddings, joint_attention_kwargs=joint_attention_kwargs)
        # 每个扩散步进行10步的优化，10步梯度下降接近最小值
        for j in range(num_inner_steps):
            noise_pred_uncond = get_noise_pred_single(pipe=pipe, latents=latent_cur, t=t, uncond_embeddings=uncond_embeddings, cond_embeddings=None, joint_attention_kwargs=joint_attention_kwargs)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latents_prev_rec = prev_step(pipe, noise_pred, t, latent_cur)
            loss = nnf.mse_loss(latents_prev_rec, latent_prev)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item = loss.item()
            if loss_item < early_stop_epsilon + i * 2e-5:
                break
        uncond_embeddings_list.append(uncond_embeddings[:1].detach())
        with torch.no_grad():
            latent_cur = get_noise_pred(pipe, latent_cur, t, uncond_embeddings, cond_embeddings, joint_attention_kwargs)
    return uncond_embeddings_list

   









