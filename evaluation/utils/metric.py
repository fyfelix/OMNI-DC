import torch


def abs_relative_difference(output, target, valid_mask=None, reduction='mean'):
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    
    # Calculate per-image metric
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n

    if reduction == 'mean':
        return abs_relative_diff.mean()
    elif reduction == 'none':
        return abs_relative_diff
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def squared_relative_difference(output, target, valid_mask=None):
    actual_output = output
    actual_target = target
    square_relative_diff = (
        torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
    )
    if valid_mask is not None:
        square_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    square_relative_diff = torch.sum(square_relative_diff, (-1, -2)) / n
    return square_relative_diff.mean()


def rmse_linear(output, target, valid_mask=None, reduction='mean'):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    
    if reduction == 'mean':
        return rmse.mean()
    elif reduction == 'none':
        return rmse
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def mae_linear(output, target, valid_mask=None, reduction='mean'):
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_diff = torch.abs(diff)
    mae = torch.sum(abs_diff, (-1, -2)) / n
    
    if reduction == 'mean':
        return mae.mean()
    elif reduction == 'none':
        return mae
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def rmse_log(output, target, valid_mask=None):
    diff = torch.log(output) - torch.log(target)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def log10(output, target, valid_mask=None):
    if valid_mask is not None:
        diff = torch.abs(
            torch.log10(output[valid_mask]) - torch.log10(target[valid_mask])
        )
    else:
        diff = torch.abs(torch.log10(output) - torch.log10(target))
    return diff.mean()


# adapt from: https://github.com/imran3180/depth-map-prediction/blob/master/main.py
def threshold_percentage(output, target, threshold_val, valid_mask=None, reduction='mean'):
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    # Use boolean comparison and float cast to keep on device
    bit_mat = (max_d1_d2 < threshold_val).float()

    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n
    
    if reduction == 'mean':
        return threshold_mat.mean()
    elif reduction == 'none':
        return threshold_mat
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def delta1_acc(pred, gt, valid_mask, reduction='mean'):
    return threshold_percentage(pred, gt, 1.25, valid_mask, reduction=reduction)


def delta2_acc(pred, gt, valid_mask, reduction='mean'):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask, reduction=reduction)


def delta3_acc(pred, gt, valid_mask, reduction='mean'):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask, reduction=reduction)

def delta4_acc_105(pred, gt, valid_mask, reduction='mean'):
    return threshold_percentage(pred, gt, 1.05, valid_mask, reduction=reduction)

def delta5_acc110(pred, gt, valid_mask, reduction='mean'):
    return threshold_percentage(pred, gt, 1.10, valid_mask, reduction=reduction)

def delta6_acc103(pred, gt, valid_mask, reduction='mean'):
    return threshold_percentage(pred, gt, 1.03, valid_mask, reduction=reduction)


def i_rmse(output, target, valid_mask=None):
    output_inv = 1.0 / output
    target_inv = 1.0 / target
    diff = output_inv - target_inv
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def silog_rmse(depth_pred, depth_gt, valid_mask=None):
    diff = torch.log(depth_pred) - torch.log(depth_gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = depth_gt.shape[-2] * depth_gt.shape[-1]

    diff2 = torch.pow(diff, 2)

    first_term = torch.sum(diff2, (-1, -2)) / n
    second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    loss = torch.sqrt(torch.mean(first_term - second_term)) * 100
    return loss
