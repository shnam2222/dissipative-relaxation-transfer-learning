import numpy as np
import torch
import torch.nn as nn
# import gc
from tqdm import tqdm

from loss_ftn import *
from timeit import default_timer

# Global parameters for plroblem setting

P = 1000
t = 500
N_L = 15
N_C = 64

dx = P/N_C
dz = t/N_L

wl_list = list(range(650, 751, 10))

# Local

PATH = "D:\\01_Datasets\\DIRTL_Manuscript\\"

def waveprior(data_dims, wavelength, dz):
    wp = np.empty((data_dims[0], data_dims[1]), np.float32)
    for i in range(data_dims[0]):
        wp[i] = np.cos(2*np.pi*i*dz/wavelength)
    return wp

def data_loader(n_train, n_test, batch_size, wl_list, data_dims, k, dz):

    train_structs = np.empty((n_train*len(wl_list), 2, data_dims[0], data_dims[1]), np.float32)
    train_results = np.empty((n_train*len(wl_list), 2, data_dims[0], data_dims[1]), np.float32)

    for i in tqdm(range(n_train), desc="Loading Train Data"):
        tmp_struct = np.load(PATH+ "patternings_multiwl\\patterning_"+f'{i:08d}'+".npy")
        tmp_result = np.load(PATH + "true_results_multiwl\\result_Hy_"+f'{i:08d}'+'k'+f'{k:.4f}'+".npy")

        train_space = np.empty(data_dims)
        train_space[:15] = np.ones((1*15,64))
        train_space[15:30] = np.repeat(np.repeat(tmp_struct, 15, axis=0), 1, axis=1)
        train_space[30:] = np.ones((5*15,64))

        for tmp, wl in enumerate(wl_list):
            train_structs[i*len(wl_list) + tmp][0] = np.copy(train_space)**2
            train_structs[i*len(wl_list) + tmp][1] = np.copy(waveprior(data_dims, wl, dz))
            train_results[i*len(wl_list) + tmp][0] = np.real(tmp_result[tmp])
            train_results[i*len(wl_list) + tmp][1] = np.imag(tmp_result[tmp])

    ## Import Data: Test
    test_structs = np.empty((n_test*len(wl_list), 2, data_dims[0], data_dims[1]), np.float32)
    test_results = np.empty((n_test*len(wl_list), 2, data_dims[0], data_dims[1]), np.float32)

    for i in tqdm(range(n_test), desc="Loading Test Data"):
        tmp_struct = np.load(PATH+ "patternings_multiwl\\patterning_"+f'{9000+i:08d}'+".npy")
        tmp_result = np.load(PATH + "true_results_multiwl\\result_Hy_"+f'{9000+i:08d}'+'k'+f'{k:.4f}'+".npy")

        test_space = np.empty(data_dims)
        test_space[:15] = np.ones((1*15,64))
        test_space[15:30] = np.repeat(np.repeat(tmp_struct, 15, axis=0), 1, axis=1)
        test_space[30:] = np.ones((5*15,64))

        for tmp, wl in enumerate(wl_list):
            test_structs[i*len(wl_list) + tmp][0] = np.copy(test_space)**2
            test_structs[i*len(wl_list) + tmp][1] = np.copy(waveprior(data_dims, wl, dz))
            test_results[i*len(wl_list) + tmp][0] = np.real(tmp_result[tmp])
            test_results[i*len(wl_list) + tmp][1] = np.imag(tmp_result[tmp])

    # Convert to torch tensors
    train_input = torch.tensor(train_structs, dtype=torch.float32)
    train_output = torch.tensor(train_results, dtype=torch.float32)

    test_input = torch.tensor(test_structs, dtype=torch.float32)
    test_output = torch.tensor(test_results, dtype=torch.float32)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_input, train_output), 
        batch_size=batch_size, shuffle=True
    )
    test_loader  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_input, test_output), 
        batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader

def data_loader_test(n_test, batch_size, wl_list, data_dims, k, dz):

    ## Import Data: Test
    test_structs = np.empty((n_test*len(wl_list), 2, data_dims[0], data_dims[1]), np.float32)
    test_results = np.empty((n_test*len(wl_list), 2, data_dims[0], data_dims[1]), np.float32)

    for i in tqdm(range(n_test), desc="Loading Test Data"):
        tmp_struct = np.load(PATH+ "patternings_multiwl_test\\patterning_"+f'{i:08d}'+".npy")
        tmp_result = np.load(PATH + "true_results_multiwl_test\\result_Hy_"+f'{i:08d}'+'k'+f'{k:.4f}'+".npy")

        test_space = np.empty(data_dims)
        test_space[:15] = np.ones((1*15,64))
        test_space[15:30] = np.repeat(np.repeat(tmp_struct, 15, axis=0), 1, axis=1)
        test_space[30:] = np.ones((5*15,64))

        for tmp, wl in enumerate(wl_list):
            test_structs[i*len(wl_list) + tmp][0] = np.copy(test_space)**2
            test_structs[i*len(wl_list) + tmp][1] = np.copy(waveprior(data_dims, wl, dz))
            test_results[i*len(wl_list) + tmp][0] = np.real(tmp_result[tmp])
            test_results[i*len(wl_list) + tmp][1] = np.imag(tmp_result[tmp])


    test_input = torch.tensor(test_structs, dtype=torch.float32)
    test_output = torch.tensor(test_results, dtype=torch.float32)

    # DataLoaders
    test_loader  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_input, test_output), 
        batch_size=batch_size, shuffle=False
    )

    return test_loader


def transfer(src, tgt, layer_pretrain):
    # ---- Transfer and freeze lift conv_in layer ----
    with torch.no_grad():
        tgt.conv_in.weight.copy_(src.conv_in.weight)
        tgt.conv_in.bias.copy_(src.conv_in.bias)

    # Freeze lift parameters
    tgt.conv_in.weight.requires_grad = False
    tgt.conv_in.bias.requires_grad = False

    # Transfer and freeze the first N FNO blocks
    for i in range(layer_pretrain):
        with torch.no_grad():
            # 1. Transfer FFT convolution weights (complex weights stored in w)
            tgt.fno_blocks[i].fftconv.w.copy_(src.fno_blocks[i].fftconv.w)

            # 2. Transfer pointwise conv weights
            tgt.fno_blocks[i].conv.weight.copy_(src.fno_blocks[i].conv.weight)

        # Freeze the copied parameters
        tgt.fno_blocks[i].fftconv.w.requires_grad = False
        tgt.fno_blocks[i].conv.weight.requires_grad = False

    return tgt