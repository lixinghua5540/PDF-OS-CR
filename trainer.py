import os

import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter

from datasets.common import set_input
from losses import discriminator_loss_func, generator_loss_func
from utils.distributed import reduce_loss_dict
from utils.metrics import PSNR, SSIM
from utils.model_utils import requires_grad


def move_to_cuda(*tensors):
    """Move tensors to CUDA."""
    return [tensor.cuda() for tensor in tensors]


def forward_generator(generator, cloudy, sar_block, sar_edge, sar):
    """Run generator with cloudy image and SAR features."""
    sar_features = torch.cat((sar_edge, sar), dim=1)
    return generator(cloudy, sar_block, sar_features)


def calculate_generator_loss(opts, output, ground_truth, sar_edge, output_pred, output_edge):
    """Calculate weighted generator loss."""
    loss_dict = generator_loss_func(
        output,
        ground_truth,
        sar_edge,
        output_pred,
        output_edge
    )

    loss_joint = (
        loss_dict["loss_adversarial"] * opts.ADVERSARIAL_LOSS
        + loss_dict["loss_l1"] * opts.L1_LOSS
        + loss_dict["loss_msssim"] * opts.MSSSIM_LOSS
    )

    loss_dict["loss_joint"] = loss_joint
    return loss_dict, loss_joint


def log_losses(writer, epoch, generator_loss_dict, discriminator_loss_dict):
    """Log training losses to TensorBoard."""
    writer.add_scalar("g_loss_adversarial", generator_loss_dict["loss_adversarial"], epoch)
    writer.add_scalar("g_loss_joint", generator_loss_dict["loss_joint"], epoch)
    writer.add_scalar("g_loss_msssim", generator_loss_dict["loss_msssim"], epoch)
    writer.add_scalar("g_loss_l1", generator_loss_dict["loss_l1"], epoch)

    writer.add_scalar("d_loss_adversarial", discriminator_loss_dict["loss_adversarial"], epoch)
    writer.add_scalar("d_loss_joint", discriminator_loss_dict["loss_joint"], epoch)


def get_reduced_losses(generator_loss_dict, discriminator_loss_dict):
    """Reduce losses for distributed training."""
    generator_loss_dict = reduce_loss_dict(generator_loss_dict)
    discriminator_loss_dict = reduce_loss_dict(discriminator_loss_dict)

    generator_losses = {
        "loss_adversarial": generator_loss_dict["loss_adversarial"].mean().item(),
        "loss_msssim": generator_loss_dict["loss_msssim"].mean().item(),
        "loss_l1": generator_loss_dict["loss_l1"].mean().item(),
        "loss_joint": generator_loss_dict["loss_joint"].mean().item(),
    }

    discriminator_losses = {
        "loss_adversarial": discriminator_loss_dict["loss_adversarial"].mean().item(),
        "loss_joint": discriminator_loss_dict["loss_joint"].mean().item(),
    }

    return generator_losses, discriminator_losses


def update_progress_bar(pbar, generator_losses, discriminator_losses):
    """Update training progress display."""
    pbar.set_description(
        f"g_loss_joint: {generator_losses['loss_joint']:.4f} "
        f"d_loss_joint: {discriminator_losses['loss_joint']:.4f} "
        f"loss_l1: {generator_losses['loss_l1']:.4f} "
        f"loss_msssim: {generator_losses['loss_msssim']:.4f} "
        f"loss_adversarial: {generator_losses['loss_adversarial']:.4f}"
    )


def save_checkpoint(save_path, epoch, generator_module, discriminator_module):
    """Save model checkpoint."""
    torch.save(
        {
            "epoch": epoch + 1,
            "generator": generator_module.state_dict(),
            "discriminator": discriminator_module.state_dict(),
        },
        save_path
    )


def validate(opts, generator, discriminator, val_loader, is_cuda):
    generator.eval()
    discriminator.eval()

    psnr_total = 0
    ssim_total = 0

    with torch.no_grad():
        for data in val_loader:
            sar, cloudy, ground_truth, sar_edge, sar_block = set_input(data)

            if is_cuda:
                sar, cloudy, ground_truth, sar_edge, sar_block = move_to_cuda(
                    sar,
                    cloudy,
                    ground_truth,
                    sar_edge,
                    sar_block
                )

            # Generate restored image.
            output = forward_generator(generator, cloudy, sar_block, sar_edge, sar)

            # Keep discriminator forward for consistent validation flow.
            discriminator(output, sar, sar_edge, is_real=False)

            psnr_total += PSNR(output, ground_truth)
            ssim_total += SSIM(output, ground_truth).item()

    return psnr_total / len(val_loader), ssim_total / len(val_loader)


def train(
    opts,
    image_data_loader,
    val_data_loader,
    generator,
    discriminator,
    generator_optim,
    discriminator_optim,
    is_cuda
):
    dataset_size = len(image_data_loader.dataset)
    print(f"Dataset size: {dataset_size}")

    if opts.distributed:
        generator_module = generator.module
        discriminator_module = discriminator.module
    else:
        generator_module = generator
        discriminator_module = discriminator

    writer = SummaryWriter(opts.log_dir)

    best_psnr_val = 0
    best_ssim_val = 0

    for epoch in range(opts.start_epoch, opts.epochs):
        print(f"Epoch {epoch + 1}/{opts.epochs}")

        generator.train()
        discriminator.train()

        pbar = tqdm(image_data_loader, dynamic_ncols=True, smoothing=0.01)

        for data in pbar:
            sar, cloudy, ground_truth, sar_edge, sar_block = set_input(data)

            if is_cuda:
                sar, cloudy, ground_truth, sar_edge, sar_block = move_to_cuda(
                    sar,
                    cloudy,
                    ground_truth,
                    sar_edge,
                    sar_block
                )

            # ----------------
            # Train generator
            # ----------------
            requires_grad(generator, True)
            requires_grad(discriminator, False)

            output = forward_generator(generator, cloudy, sar_block, sar_edge, sar)
            output_pred, output_edge = discriminator(output, sar, sar_edge, is_real=False)

            generator_loss_dict, generator_loss = calculate_generator_loss(
                opts,
                output,
                ground_truth,
                sar_edge,
                output_pred,
                output_edge
            )

            generator_optim.zero_grad()
            generator_loss.backward()
            generator_optim.step()

            # ---------------------
            # Train discriminator
            # ---------------------
            requires_grad(generator, False)
            requires_grad(discriminator, True)

            real_pred, real_pred_edge = discriminator(
                ground_truth,
                sar,
                sar_edge,
                is_real=True
            )

            fake_pred, fake_pred_edge = discriminator(
                output.detach(),
                sar,
                sar_edge,
                is_real=False
            )

            discriminator_loss_dict = discriminator_loss_func(
                real_pred,
                fake_pred,
                real_pred_edge,
                fake_pred_edge,
                sar_edge
            )

            discriminator_loss = discriminator_loss_dict["loss_adversarial"]
            discriminator_loss_dict["loss_joint"] = discriminator_loss

            discriminator_optim.zero_grad()
            discriminator_loss.backward()
            discriminator_optim.step()

            # ----------------
            # Log train losses
            # ----------------
            generator_losses, discriminator_losses = get_reduced_losses(
                generator_loss_dict,
                discriminator_loss_dict
            )

            update_progress_bar(pbar, generator_losses, discriminator_losses)
            log_losses(writer, epoch, generator_losses, discriminator_losses)

        # ----------------
        # Validate model
        # ----------------
        psnr_val, ssim_val = validate(
            opts,
            generator,
            discriminator,
            val_data_loader,
            is_cuda
        )

        print(
            f"Epoch {epoch + 1}, "
            f"PSNR Val: {psnr_val:.4f}, "
            f"SSIM Val: {ssim_val:.4f}"
        )

        # Save the best model.
        if psnr_val > best_psnr_val and ssim_val > best_ssim_val:
            best_psnr_val = psnr_val
            best_ssim_val = ssim_val

            print(
                f"Saving best model "
                f"(best_PSNR={best_psnr_val:.4f}) "
                f"(best_SSIM={best_ssim_val:.4f})..."
            )

            save_checkpoint(
                os.path.join(opts.save_dir, "best_model.pt"),
                epoch,
                generator_module,
                discriminator_module
            )

        # Save checkpoint periodically.
        if (epoch + 1) % opts.save_epoch == 0 and epoch >=30:
            checkpoint_name = f"{str(epoch + 1).zfill(6)}.pt"

            save_checkpoint(
                os.path.join(opts.save_dir, checkpoint_name),
                epoch,
                generator_module,
                discriminator_module
            )

    writer.close()
