import torch.nn as nn

from pytorch_msssim import ms_ssim


def generator_loss_func(output, ground_truth, edge, output_pred, output_edge):

    l1 = nn.L1Loss()
    criterion = nn.BCELoss()

    loss_l1 = l1(output, ground_truth)

    real_target = output_pred.new_ones(output_pred.shape)
    loss_adversarial = criterion(output_pred, real_target) + criterion(output_edge, edge)

    loss_msssim = 1 - ms_ssim(output, ground_truth, data_range=1.0, size_average=True)

    return {
        'loss_l1': loss_l1.mean(),
        'loss_adversarial': loss_adversarial.mean(),
        'loss_msssim': loss_msssim.mean()
    }


def discriminator_loss_func(real_pred, fake_pred, real_pred_edge, fake_pred_edge, edge):

    criterion = nn.BCELoss()
    
    real_target = real_pred.new_ones(real_pred.shape)
    fake_target = fake_pred.new_zeros(fake_pred.shape)

    loss_adversarial = criterion(real_pred, real_target) + criterion(fake_pred, fake_target) + \
                    criterion(real_pred_edge, edge) + criterion(fake_pred_edge, edge)

    return {
        'loss_adversarial': loss_adversarial.mean()
    }
