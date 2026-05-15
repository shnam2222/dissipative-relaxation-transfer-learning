import torch
import torch.nn as nn
import numpy as np


def rel_err(model, data_loader, filename=None, device='cuda'):
    """
    Accelerated error computation using DataLoader batching.
    """
    model.eval()
    model.to(device)

    # Preallocate lists
    L1, L2, rel_L1, rel_L2, max_amplitudes = [], [], [], [], []

    with torch.no_grad():
        for batch_structs, batch_results in data_loader:
            batch_structs = batch_structs.to(device)
            batch_results = batch_results.to(device)

            # Forward pass
            preds = model(batch_structs)  # shape: [B, 2, ...]
            
            # Convert to complex numbers
            pred_complex = preds[:,0] + 1j*preds[:,1]
            true_complex = batch_results[:,0] + 1j*batch_results[:,1]

            # Vectorized computation across batch
            abs_true = torch.abs(true_complex)
            abs_diff = torch.abs(pred_complex - true_complex)

            # Compute metrics per sample
            sum_abs_diff = torch.sum(abs_diff, dim=list(range(1, abs_diff.ndim)))
            sum_abs_true = torch.sum(abs_true, dim=list(range(1, abs_true.ndim)))

            L1_batch = sum_abs_diff.cpu().numpy()
            L2_batch = torch.sqrt(torch.sum(abs_diff**2, dim=list(range(1, abs_diff.ndim)))).cpu().numpy()
            rel_L1_batch = (sum_abs_diff / sum_abs_true).cpu().numpy()
            rel_L2_batch = (L2_batch / torch.sqrt(torch.sum(abs_true**2, dim=list(range(1, abs_true.ndim)))).cpu()).numpy()
            max_amp_batch = torch.amax(abs_true, dim=list(range(1, abs_true.ndim))).cpu().numpy()

            L1.extend(L1_batch)
            L2.extend(L2_batch)
            rel_L1.extend(rel_L1_batch)
            rel_L2.extend(rel_L2_batch)
            max_amplitudes.extend(max_amp_batch)

    # Save once outside loop
    if filename:
        np.savez_compressed(
            filename,
            L1=np.array(L1),
            L2=np.array(L2),
            rel_L1=np.array(rel_L1),
            rel_L2=np.array(rel_L2),
            max_amplitudes=np.array(max_amplitudes)
        )

    return L1, L2, rel_L1, rel_L2, max_amplitudes


# From J.Fan WaveYNet

def eps_x(eps):
    eps_ik_1 = torch.roll(eps, shifts = (0, 0, 1), dims = (0, 1, 2))
    eps_ik_1[0, :] = eps[0,:]
    return (eps + eps_ik_1)/2

def eps_z(eps):
    eps_i_1k = torch.roll(eps, shifts = (0, 0, 1), dims = (0, 1, 2))
    return (eps + eps_i_1k)/2

def maxwell_eq_H(H, eps, wl, dx, dz):
    
    """
    Batch-compatible Maxwell residual computation
    H: complex tensor of shape (B, X, Z)
    device_struct: permittivity tensor (B, X, Z)
    wl: tensor of shape (B,)
    dx, dz: scalar grid spacings
    """

    # Local copies
    H_ik1 = torch.zeros_like(H)
    H_ik1[:, :-1, :] = H[:, 1:, :]
    H_ik_1 = torch.zeros_like(H)
    H_ik_1[:, 1:, :] = H[:, :-1, :]

    H_i1k = torch.zeros_like(H)
    H_i1k[:, :, :-1] = H[:, :, 1:]
    H_i_1k = torch.zeros_like(H)
    H_i_1k[:, :, 1:] = H[:, :, :-1]

    eps_x_ik = eps_x(eps)
    eps_z_ik = eps_z(eps)
    eps_x_ik1 = torch.roll(eps_x_ik, shifts=-1, dims=1)
    eps_x_ik1[:, -1, :] = 1.0
    eps_z_i1k = torch.roll(eps_z_ik, shifts=-1, dims=2)
    eps_z_i1k[:, :, -1] = 1.0

    # Maxwell residual
    res = (
        ((H - H_ik1) / eps_x_ik1 + (H - H_ik_1) / eps_x_ik) / (dz**2)
        - ((H_i1k - H) / eps_z_i1k + (-H + H_i_1k) / eps_z_ik) / (dx**2)
    )

    # wavelength scaling — broadcast to all spatial dims
    const = (wl.view(-1, 1, 1)**2) / (4 * np.pi**2)
    residual_H = const * res
    return residual_H


def maxwellloss(H_complex, device_struct, wl, dx, dz):
    """
    Maxwell loss for batched complex fields
    H_complex: (B, X, Z), complex-valued
    device_struct: (B, X, Z)
    wl: (B,)
    """
    res_H = maxwell_eq_H(H_complex, device_struct, wl, dx, dz)
    b = H_complex.size(0)

    H_r = H_complex.real[:, 1:-1, 1:-1]
    H_i = H_complex.imag[:, 1:-1, 1:-1]
    res_H_r = res_H.real[:, 1:-1, 1:-1]
    res_H_i = res_H.imag[:, 1:-1, 1:-1]

    loss = nn.MSELoss(reduction='mean')
    loss_r = loss(H_r.reshape(b, -1), res_H_r.reshape(b, -1))
    loss_i = loss(H_i.reshape(b, -1), res_H_i.reshape(b, -1))
    return loss_r + loss_i
