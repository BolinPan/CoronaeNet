import numpy as np
import math
import os
import hdf5storage as h5
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset




"""
Define data loader for UNet denoiser
"""

class datasetUNet(Dataset):
    """
    Loads images and labels.
        folder_im        (str): path to the folder containing the images
        file_names_im   (list): list of strings, list of names of images
        file_list_im    (list): list of strings, paths to images
        folder_label     (str): path to the folder containing the labels
        file_names_label(list): list of strings, list of names of labels
        file_list_label (list): list of strings, paths to labels
    """
    def __init__(self, folder):
        """
        Loads images before feeding it to the network.
        """
        assert os.path.exists(folder), 'Images folder does not exist'

        super(datasetUNet, self).__init__()

        self.folder_im = os.path.join(folder, 'Images')
        im_list = next(os.walk(self.folder_im))[2]
        im_list.sort(key=lambda x: int(x.split('_')[1][:-4]))
        self.file_names_im = im_list
        self.file_list_im = [os.path.join(self.folder_im, i) for i in self.file_names_im]

        self.folder_label = os.path.join(folder, 'Labels')
        label_list = next(os.walk(self.folder_label))[2]
        label_list.sort(key=lambda x: int(x.split('_')[1][:-4]))
        self.file_names_label = label_list
        self.file_list_label = [os.path.join(self.folder_label, i) for i in self.file_names_label]

    def __getitem__(self, index):
        """
        Loads and transforms an image.
        Inputs:
            index (int): index of the image in the list of files, should point to .mat
        Returns:
            imF     (torch.Tensor): finest scale coefficient image
            imSF    (torch.Tensor): semi-finest scale coefficient image
            imC     (torch.Tensor): coarse scale coefficient image
            labelF  (torch.Tensor): finest scale coefficient label
            labelSF (torch.Tensor): semi-finest scale coefficient label
            labelC  (torch.Tensor): coarse scale coefficient label
        """
        # load images
        im = torch.from_numpy(h5.loadmat(self.file_list_im[index])['im']).float()

        # load labels
        label = torch.from_numpy(h5.loadmat(self.file_list_label[index])['label']).float()
        return im, label

    def __len__(self):
        return len(self.file_list_im)




def LoadDataUNet(path_datasets, mode, minibatch_size):
    """
    Load dataset for learning
    Inputs:
        path_datasets  (str): path to dataset folder
        mode           (str): select train, validate or test
        minibatch_size (int): batch size
    Returns:
        loader: data loader
    """
    if mode == 'train':
        # training set
        path_train = os.path.join(path_datasets,'train')
        train_data = datasetUNet(folder = path_train)
        train_loader = DataLoader(train_data, batch_size = minibatch_size, shuffle=True)
        loader = train_loader
    elif mode == 'test':
        # test set
        path_test = os.path.join(path_datasets,'test')
        test_data = datasetUNet(folder = path_test)
        test_loader = DataLoader(test_data, batch_size = minibatch_size,shuffle=False)
        loader = test_loader
    elif mode == 'val':
        # validation set
        path_val = os.path.join(path_datasets,'val')
        val_data = datasetUNet(folder=path_val)
        val_loader = DataLoader(val_data, batch_size = minibatch_size, shuffle=True)
        loader = val_loader
    return loader






"""
Define data loader for MSCNet 
"""

class dataset(Dataset):
    """
    Loads images and labels.
        folder_im        (str): path to the folder containing the images
        file_names_im   (list): list of strings, list of names of images
        file_list_im    (list): list of strings, paths to images
        folder_label     (str): path to the folder containing the labels
        file_names_label(list): list of strings, list of names of labels
        file_list_label (list): list of strings, paths to labels
    """
    def __init__(self, folder):
        """
        Loads images before feeding it to the network.
        """
        assert os.path.exists(folder), 'Images folder does not exist'

        super(dataset, self).__init__()

        self.folder_im = os.path.join(folder, 'Images')
        im_list = next(os.walk(self.folder_im))[2]
        im_list.sort(key=lambda x: int(x.split('_')[1][:-4]))
        self.file_names_im = im_list
        self.file_list_im = [os.path.join(self.folder_im, i) for i in self.file_names_im]

        self.folder_label = os.path.join(folder, 'Labels')
        label_list = next(os.walk(self.folder_label))[2]
        label_list.sort(key=lambda x: int(x.split('_')[1][:-4]))
        self.file_names_label = label_list
        self.file_list_label = [os.path.join(self.folder_label, i) for i in self.file_names_label]

    def __getitem__(self, index):
        """
        Loads and transforms an image.
        Inputs:
            index (int): index of the image in the list of files, should point to .mat
        Returns:
            imF     (torch.Tensor): finest scale coefficient image
            imSF    (torch.Tensor): semi-finest scale coefficient image
            imC     (torch.Tensor): coarse scale coefficient image
            labelF  (torch.Tensor): finest scale coefficient label
            labelSF (torch.Tensor): semi-finest scale coefficient label
            labelC  (torch.Tensor): coarse scale coefficient label
        """
        # load images
        imF = torch.from_numpy(h5.loadmat(self.file_list_im[index])['imF']).float()
        imSF = torch.from_numpy(h5.loadmat(self.file_list_im[index])['imSF']).float()
        imC = torch.from_numpy(h5.loadmat(self.file_list_im[index])['imC']).float()

        # load labels
        label = torch.from_numpy(h5.loadmat(self.file_list_label[index])['label']).float()
        return imF, imSF, imC, label
    def __len__(self):
        return len(self.file_list_im)




def LoadData(path_datasets, mode, minibatch_size):
    """
    Load dataset for learning
    Inputs:
        path_datasets  (str): path to dataset folder
        mode           (str): select train, validate or test
        minibatch_size (int): batch size
    Returns:
        loader: data loader
    """
    if mode == 'train':
        # training set
        path_train = os.path.join(path_datasets,'train')
        train_data = dataset(folder = path_train)
        train_loader = DataLoader(train_data, batch_size = minibatch_size, shuffle=True)
        loader = train_loader
    elif mode == 'test':
        # test set
        path_test = os.path.join(path_datasets,'test')
        test_data = dataset(folder = path_test)
        test_loader = DataLoader(test_data, batch_size = minibatch_size,shuffle=False)
        loader = test_loader
    elif mode == 'val':
        # validation set
        path_val = os.path.join(path_datasets,'val')
        val_data = dataset(folder=path_val)
        val_loader = DataLoader(val_data, batch_size = minibatch_size, shuffle=True)
        loader = val_loader
    return loader




# define mse loss
class Loss_l2:
    def __init__(self):
        self.criterion = nn.MSELoss()

    def __call__(self, output, label):
        criterion = self.criterion(output, label)
        return criterion




"""
****************  Define Loss of MSCurveNet  ****************
"""
class Loss_l2_Inv:
    def __init__(self, weights):

        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.MSELoss()
        self.criterion3 = nn.MSELoss()
        self.weights = weights

    def __call__(self, output, target):
        criterion = (
                self.weights[0] * self.criterion1(output[:, 1, :, :], target[:, 1, :, :])
                + self.weights[1] * self.criterion2(output[:, 3, :, :], target[:, 3, :, :])
                + self.weights[2] * self.criterion3(output[:, 5, :, :], target[:, 5, :, :])
        )
        return criterion


class Loss_Mix_Inv:
    def __init__(self, weights):

        self.criterion1 = nn.L1Loss()
        self.criterion2 = nn.L1Loss()
        self.criterion3 = nn.MSELoss()
        self.weights = weights

    def __call__(self, output, target):

        criterion = (
                self.weights[0] * self.criterion1(output[:,1,:,:], target[:,1,:,:])
                + self.weights[1] * self.criterion2(output[:,3,:,:], target[:,3,:,:])
                + self.weights[2] * self.criterion3(output[:,5,:,:], target[:,5,:,:])
        )

        return criterion


class Loss_l2_All:
    def __init__(self, weights):

        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.MSELoss()
        self.criterion3 = nn.MSELoss()
        self.weights = weights

    def __call__(self, output, target):

        criterion = (
                self.weights[0]*self.criterion1(output[:,0,:,:], target[:,0,:,:])
                + self.weights[0]*self.criterion1(output[:,1,:,:], target[:,1,:,:])
                + self.weights[1]*self.criterion2(output[:,2,:,:], target[:,2,:,:])
                + self.weights[1]*self.criterion2(output[:,3,:,:], target[:,3,:,:])
                + self.weights[2]*self.criterion3(output[:,4,:,:], target[:,4,:,:])
                + self.weights[2]*self.criterion3(output[:,5,:,:], target[:,5,:,:])
        )

        return criterion


# double convolution
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))




def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim=dim, start=0, length=x.size(dim) - shift)
    right = x.narrow(dim=dim, start=x.size(dim) - shift, length=shift)
    return torch.cat((right, left), dim=dim)




def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)




def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)




def rfft2(data):
    data = ifftshift(data, dim=(-2, -1))
    data = torch.rfft(data, 2, normalized=True, onesided=False)
    data = fftshift(data, dim=(-3, -2))
    return data




def irfft2(data):
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.irfft(data, 2, normalized=True, onesided=False)
    data = fftshift(data, dim=(-2, -1))
    return data




class Initializer:
    """
    This is a class to make initializing the weightes easier in pytorch
    github: https://github.com/3ammor/Weights-Initializer-pytorch
    """
    def __init__(self):
        pass

    @staticmethod
    def initialize(model, initialization, **kwargs):

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.Linear):
                initialization(m.weight.data, **kwargs)
                try:
                    initialization(m.bias.data)
                except:
                    pass

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.0)
                m.bias.data.fill_(0)

        model.apply(weights_init)





class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    github: https://github.com/Bjarten/early-stopping-pytorch
    MIT License
    Copyright(c) 2018 Bjarte Mehus Sunde
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func("EarlyStopping counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func("Validation loss decreased ({} --> {}).  Saving model ...".format('%6f' % self.val_loss_min, '%6f' % val_loss))
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss




def fdct_wrapping_window(x):
    """
    This is numpy implementation of creating the two halves of a
    C^inf compactly supported window, see fdct_wrapping_window.m
    Inputs:
        x (numpy.array): vector or matrix of abscissae, the relevant ones from 0 to 1
    Returns:
        numpy.array: vector or matrix containing samples of the left, resp. right
        half of the window

    This source code is taken from the CurveLab toolbox implemented in Matlab
    http://www.curvelet.org/software.html
    By Laurent Demanet, 2004
    Modified by Bolin Pan, March 2021
    """
    wr = np.zeros(x.shape)
    wl = np.zeros(x.shape)
    x[np.abs(x) < 2 ** (-52)] = 0
    wr[(x > 0) * (x < 1)] = np.exp((1 - 1 / (1 - np.exp((1 - 1 / x[(x > 0) * (x < 1)])))))
    wr[x <= 0] = 1
    wl[(x > 0) * (x < 1)] = np.exp((1 - 1 / (1 - np.exp((1 - 1 / (1 - x[(x > 0) * (x < 1)]))))))
    wl[x >= 1] = 1
    normalization = np.sqrt(wl ** 2 + wr ** 2)
    wr = wr / normalization
    wl = wl / normalization
    return wl, wr




def getFirstLowpass(N1, N2, batch_size):
    """
    This is the numpy implementation to generate the first lowpass filter
    used in coronae base Curvelet decomposition in finest/semi-finest scale.
    Inputs:
        N1, N2: size of original image
    Returns:
        torch.Tensor: lowpass_first, equiv_index_1, equiv_index_2

    This source code is taken from the CurveLab toolbox implemented in Matlab
    http://www.curvelet.org/software.html
    By Laurent Demanet, 2004
    Modified by Bolin Pan, March 2021
    """
    # Initialization
    M1 = N1 / 3.0
    M2 = N2 / 3.0
    # Invariant: equiv_index_1(floor(2 * M1) + 1) == (N1 + 2 - mod(N1, 2)) / 2
    window_length_1 = math.floor(2 * M1) - math.floor(M1) - 1 - ((N1 % 3) == 0)
    window_length_2 = math.floor(2 * M2) - math.floor(M2) - 1 - ((N2 % 3) == 0)
    # Invariant: floor(M1) + floor(2 * M1) == N1 - (mod(N1, 3)~=0)
    coord_1 = np.linspace(0, 1, window_length_1 + 1)
    coord_2 = np.linspace(0, 1, window_length_2 + 1)
    wl_1, wr_1 = fdct_wrapping_window(coord_1)
    wl_2, wr_2 = fdct_wrapping_window(coord_2)
    # compute first lowpass filter
    lowpass_1 = np.concatenate((wl_1, np.ones(2 * math.floor(M1) + 1), wr_1))
    if np.mod(N1, 3) == 0:
        lowpass_1 = np.concatenate((np.zeros(1), lowpass_1, np.zeros(1)))
    lowpass_2 = np.concatenate((wl_2, np.ones(2 * math.floor(M2) + 1), wr_2))
    if np.mod(N2, 3) == 0:
        lowpass_2 = np.concatenate((np.zeros(1), lowpass_2, np.zeros(1)))

    # vectorize to compute the transpose multiplication
    lowpass_1 = lowpass_1.reshape(-1, 1)  # column vector
    lowpass_2 = lowpass_2.reshape(1, -1)  # row vector
    lowpass_first = torch.from_numpy(np.dot(lowpass_1, lowpass_2)).to(torch.float32)
    lowpass_first = lowpass_first.unsqueeze(0).unsqueeze(3)
    lowpass_first = lowpass_first.repeat(batch_size,1,1,2)
    return lowpass_first




def getHipassLowpass(M1, M2, batch_size):
    """
    This is the numpy implementation to generate the hipass and lowpass
    filters used in coronae decomposition based on Curvelet tilling in
    finest/semi-finest scale.
    Inputs:
        M1, M2 size of defined filter
    Returns:
        torch.Tensor: hipass, lowpass
    """
    # Initialization: smooth periodic extension of high frequencies
    window_length_1 = math.floor(2 * M1) - math.floor(M1) - 1
    window_length_2 = math.floor(2 * M2) - math.floor(M2) - 1
    coord_1 = np.linspace(0, 1, window_length_1 + 1)
    coord_2 = np.linspace(0, 1, window_length_2 + 1)
    wl_1, wr_1 = fdct_wrapping_window(coord_1)
    wl_2, wr_2 = fdct_wrapping_window(coord_2)
    lowpass_1 = np.concatenate((wl_1, np.ones(2 * math.floor(M1) + 1), wr_1))
    lowpass_2 = np.concatenate((wl_2, np.ones(2 * math.floor(M2) + 1), wr_2))

    # vectorize to compute the transpose multiplication
    lowpass_1 = lowpass_1.reshape(-1, 1)  # column vector
    lowpass_2 = lowpass_2.reshape(1, -1)  # row vector
    lowpass = np.dot(lowpass_1, lowpass_2)
    hipass = np.sqrt(1.0 - lowpass ** 2)

    # process filters
    hipass = torch.from_numpy(hipass).to(torch.float32)
    hipass = hipass.unsqueeze(0).unsqueeze(3)
    hipass = hipass.repeat(batch_size,1,1,2)

    lowpass = torch.from_numpy(lowpass).to(torch.float32)
    lowpass = lowpass.unsqueeze(0).unsqueeze(3)
    lowpass = lowpass.repeat(batch_size,1,1,2)
    return hipass, lowpass




def extractFinestComponents(Xlow, M1, M2, lowpass_first, hipass, lowpass):
    """
    This is the implementation to extract the hipass and lowpass components
    for coronae decomposition based on Curvelet tilling in finest scale.
    Inputs:
        (torch.Tensor) Xlow: input unfold (lowpass first applied) fourier image
         M1, M2: size of defined filters
        (torch.Tensor) lowpass_first, hipass, lowpass: lowpass_first, hipass and lowpass filters
    Returns:
        torch.Tensor: Xhi, Xlow
    """
    # process hi and low components
    Xhi = Xlow
    # matlab Xlow_index_1 is [18,19,...,50] but here [17,18,...,49]
    X_index_1 = np.arange(-math.floor(2.0 * M1), math.floor(2.0 * M1) + 1) + math.floor(4.0 * M1)
    X_index_2 = np.arange(-math.floor(2.0 * M2), math.floor(2.0 * M2) + 1) + math.floor(4.0 * M2)
    # change data type, -1 is not necessary here: because we pretend miss +1 at the above lines
    X_index_1 = X_index_1.astype(int)
    X_index_2 = X_index_2.astype(int)

    # follow extraction from fdct_wrapping.m
    Xlow = Xlow[:,:,X_index_2,:][:,X_index_1,:,:]
    Xhi[:,:,X_index_2[0]:X_index_2[-1]+1,:][:,X_index_1[0]:X_index_1[-1]+1,:,:] = Xlow * hipass
    Xlow = Xlow * lowpass  # size is 2*floor(2*M1)+1 - by - 2*floor(2*M2)+1

    # follow extraction from ifdct_wrapping.m
    Xhi = Xhi*lowpass_first
    Xhi[:,:,X_index_2[0]:X_index_2[-1]+1,:][:,X_index_1[0]:X_index_1[-1]+1,:,:] = \
        Xhi[:,:,X_index_2[0]:X_index_2[-1]+1,:][:,X_index_1[0]:X_index_1[-1]+1,:,:] * hipass
    Xlow = Xlow * lowpass
    return Xhi, Xlow



def extractSemiFinestComponents(Xlow, M1, M2, hipass, lowpass):
    """
    This is the implementation to extract the hipass and lowpass components
    for coronae decomposition based on Curvelet tilling in semi-finest scale
    Inputs:
        (torch.Tensor) Xlow: input unfold (lowpass first applied) fourier image
         M1, M2: size of defined filters
        (torch.Tensor) lowpass_first, hipass, lowpass: lowpass_first, hipass and lowpass filters
    Returns:
        torch.Tensor: Xhi, Xlow
    """
    # process hi and low components
    Xhi = Xlow
    # matlab Xlow_index_1 is [18,19,...,50] but here [17,18,...,49]
    X_index_1 = np.arange(-math.floor(2.0* M1), math.floor(2.0 * M1) + 1) + math.floor(4.0 * M1)
    X_index_2 = np.arange(-math.floor(2.0* M2), math.floor(2.0 * M2) + 1) + math.floor(4.0 * M2)
    # change data type, -1 is not necessary here: because we pretend miss +1 at the above lines
    X_index_1 = X_index_1.astype(int)
    X_index_2 = X_index_2.astype(int)

    # follow extraction from fdct_wrapping.m
    Xlow = Xlow[:,:,X_index_2,:][:,X_index_1,:,:]
    Xhi[:,:,X_index_2[0]:X_index_2[-1]+1,:][:,X_index_1[0]:X_index_1[-1]+1,:,:] = Xlow * hipass
    Xlow = Xlow * lowpass  # size is 2*floor(2*M1)+1 - by - 2*floor(2*M2)+1

    # follow extraction from ifdct_wrapping.m
    Xhi[:,:,X_index_2[0]:X_index_2[-1]+1,:][:,X_index_1[0]:X_index_1[-1]+1,:,:] = \
        Xhi[:,:,X_index_2[0]:X_index_2[-1]+1,:][:,X_index_1[0]:X_index_1[-1]+1, :,:] * hipass
    Xlow = Xlow * lowpass
    return Xhi, Xlow




def wrapping(X, N1, N2):
    """
    This is the numpy implementation to wrap the out-of-support image
    in fourier domain into the original image support
    Inputs:
        (torch.Tensor) X: input image in fourier domain
        N1, N2: size of original image
    Returns:
        torch.Tensor: wrapped image in fourier domain
    """
    M1 = N1 / 3
    M2 = N2 / 3
    shift_1 = math.floor(2 * M1) - math.floor(N1 / 2)
    shift_2 = math.floor(2 * M2) - math.floor(N2 / 2)

    Y = X[:,:,np.arange(1, N2+ 1) + shift_2 - 1,:]
    Y[:,:,N2 - shift_2 + np.arange(1, shift_2 + 1) - 1,:] = Y[:, :,N2 - shift_2 + np.arange(1, shift_2 + 1) - 1,:] + \
                                                         X[:, :,np.arange(1, shift_2 + 1) - 1,:]
    Y[:, :,np.arange(1, shift_2 + 1) - 1,:] = Y[:, :,np.arange(1, shift_2 + 1) - 1,:] + X[:,:,N2 + shift_2 + np.arange(1, shift_2 + 1) - 1,:]
    X = Y[:,np.arange(1, N1 + 1) + shift_1 - 1, :,:]
    X[:,N1 - shift_1 + np.arange(1, shift_1 + 1) - 1, :,:] = X[:,N1 - shift_1 + np.arange(1, shift_1 + 1) - 1, :,:] + \
                                                         Y[:,np.arange(1, shift_1 + 1) - 1, :,:]
    X[:,np.arange(1, shift_1 + 1) - 1, :,:] = X[:,np.arange(1, shift_1 + 1) - 1, :,:] + Y[:,(N1 + shift_1 + np.arange(1, shift_1 + 1) - 1), :,:]
    return X




def padLowpassComponent(Xlow, S1, S2):
    """
    This is the implementation of zero padding lowpass component to size S1 x S2
    Inputs:
        (torch.Tensor) Xlow: lowpass component
        S1, S2: size of padded image
    Returns:
        torch.Tensor: Xlow_pad
        """
    # compute offset for padding
    n1 = Xlow.shape[1]
    n2 = Xlow.shape[2]
    off1 = int((S1 - n1) / 2)
    off2 = int((S2 - n2) / 2)

    # pad low component to original size
    padding = nn.ZeroPad2d((off1, off2, off1, off2))
    Xlow_pad = torch.squeeze(padding(Xlow.transpose_(1, 3)).transpose_(1, 3))
    return Xlow_pad


