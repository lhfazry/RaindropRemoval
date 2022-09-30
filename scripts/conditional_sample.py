"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
from guided_diffusion import script_util

import numpy as np
import torch as th
import torch.distributed as dist
import guided_diffusion.script_util
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        random_flip=False,
        deterministic=True
    )

    logger.log("sampling...")

    #for step, images in enumerate(data):
        #x = 0

    all_images = []
    all_labels = []
    all_origins = []
    all_noises = []
    #while len(all_images) * args.batch_size < args.num_samples:
    for step, (images, cond) in enumerate(data):
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        images = images.to(dist_util.dev())
        noise = th.randn_like(images)
        #t, _ = schedule_sampler.sample(noise.shape[0], dist_util.dev())
        t = (args.from_noise_step * th.ones(noise.shape[0]).long()).to(dist_util.dev())
        x_t = diffusion.q_sample(images, t, noise=noise)

        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            noise=x_t,
            from_noise_step=args.from_noise_step,
            clip_denoised=args.clip_denoised,
            progress=True,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        all_noises.extend([item for item in x_t])
        all_origins.extend([item for item in images])

        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

        if len(all_images) * args.batch_size >= args.num_samples:
            break
    
    #np.savez(os.path.join(logger.get_dir(), f"noises.npz"), np.concatenate(all_noises, axis=0))
    np.savez(os.path.join(logger.get_dir(), f"noises.npz"), script_util.tensors_to_images(th.stack(all_noises)).cpu().numpy())
    np.savez(os.path.join(logger.get_dir(), f"origins.npz"), script_util.tensors_to_images(th.stack(all_origins)).cpu().numpy())

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        from_noise_step=1000,
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        schedule_sampler="uniform",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
