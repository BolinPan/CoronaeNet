import torch
import numpy as np
import math

from torch.autograd import Function
from util import rfft2, irfft2
from util import wrapping, padLowpassComponent
from util import extractFinestComponents, extractSemiFinestComponents








class finestCoronaeDecLayer(Function):
    """
    Implement forward backward Coronae base Curvelet decomposition in finest scale.
    Inputs:
        x (torch.Tensor): input images of size (N, C, H, W)
        lowpass_first (torch.Tensor): first lowpass filter
        hipasss and lowpass (torch.Tensor): hipass and lowpass filters for finest scale
    Returns:
        torch Tensors: imHi, imLow
    """

    @staticmethod
    def forward(ctx, x, lowpass_first, hipass, lowpass):

        # save for backward
        ctx.shape = x.shape
        ctx.N1 = ctx.shape[2]
        ctx.N2 = ctx.shape[3]
        ctx.lowpass_first = lowpass_first
        ctx.hipass = hipass
        ctx.lowpass = lowpass

        # Initialization: smooth periodic extension of high frequencies
        M1 = ctx.N1 / 3
        M2 = ctx.N2 / 3
        bigN1 = 2 * math.floor(2 * M1) + 1
        bigN2 = 2 * math.floor(2 * M2) + 1
        equiv_index_1 = 1 + np.mod(math.floor(ctx.N1 / 2) - math.floor(2 * M1) + np.arange(1, bigN1 + 1) - 1, ctx.N1)
        equiv_index_2 = 1 + np.mod(math.floor(ctx.N2 / 2) - math.floor(2 * M1) + np.arange(1, bigN2 + 1) - 1, ctx.N2)

        # change type and index starting at [0]
        equiv_index_1 = equiv_index_1.astype(int) - 1
        equiv_index_2 = equiv_index_2.astype(int) - 1

        # udpate M1, M2 for later used
        M1 = M1 / 2
        M2 = M2 / 2

        # reshape input from 4D to 3D
        depth = ctx.shape[0] * ctx.shape[1]
        x_3d = torch.reshape(x, [depth, ctx.shape[2], ctx.shape[3]])

        # forward FFT
        X = rfft2(x_3d)
        Xlow = X[:,:,equiv_index_2,:][:,equiv_index_1,:,:] * ctx.lowpass_first

        # extract hipass and lowpass components
        Xhi, Xlow = extractFinestComponents(Xlow, M1, M2, lowpass_first, hipass, lowpass)

        # folding Xhi back onto N1 x N2 matrix
        Xhi = wrapping(Xhi, ctx.N1, ctx.N2)

        # inverse unitary FFT
        imHi = irfft2(Xhi)
        imLow = irfft2(Xlow)

        # reshape components from 3D to 4D then output
        output_imHi = imHi.view(ctx.shape[0], ctx.shape[1], imHi.shape[1], imHi.shape[2])
        output_imLow = imLow.view(ctx.shape[0], ctx.shape[1], imLow.shape[1], imLow.shape[2])
        return output_imHi, output_imLow


    @staticmethod
    def backward(ctx, output_imHi, output_imLow):
        # reshape inputs from 4D to 3D
        depth = ctx.shape[0] * ctx.shape[1]
        imHi = torch.reshape(output_imHi, [depth, output_imHi.shape[2], output_imHi.shape[3]])
        imLow = torch.reshape(output_imLow, [depth, output_imLow.shape[2], output_imLow.shape[3]])

        # compute bigN1 and bigN2 for padding used
        M1 = ctx.N1 / 3
        M2 = ctx.N2 / 3
        bigN1 = 2 * math.floor(2 * M1) + 1
        bigN2 = 2 * math.floor(2 * M2) + 1

        # recover original image
        Xhi = rfft2(imHi)
        Xlow = rfft2(imLow)

        # zero pad lowpass component to size bigN1 x bigN2
        Xlow_pad = padLowpassComponent(Xlow, bigN1, bigN2)

        # fold padded Xlow onto N1 x N2
        Xlow = wrapping(Xlow_pad, ctx.N1, ctx.N2)

        # combine hipass and lowpass components
        X = Xhi + Xlow

        # inverse unitary FFT
        x_rec = irfft2(X)

        # reshape reconstruction from 3D to 4D
        output_x = torch.reshape(x_rec, [ctx.shape[0], ctx.shape[1], ctx.N1, ctx.N2])
        return output_x, None, None, None





class semiFinestCoronaeDecLayer(Function):
    """
    Implement forward backward Coronae based Curvelet decomposition in semi-finest scale.
    Inputs:
        x (torch.Tensor): input images of size (N, C, H, W)
        N1: original image size along x
        N2: original image size along y
        hipass, lowpass (torch.Tensor): hipass and lowpass filters for semi-finest scale
    Returns:
        torch Tensors: imLow, imHi
    """

    @staticmethod
    def forward(ctx, x, N1, N2, hipass, lowpass):
        # save for backward
        ctx.shape = x.shape
        ctx.N1 = N1
        ctx.N2 = N2
        ctx.hipass = hipass
        ctx.lowpass = lowpass

        # reshape x from 4D to 3D
        depth = ctx.shape[0] * ctx.shape[1]
        x_3d = torch.reshape(x, [depth, ctx.shape[2], ctx.shape[3]])

        # compute M1 and M2
        M1 = ctx.N1 / 3
        M2 = ctx.N2 / 3
        M1 = M1 / (2 ** 2)
        M2 = M2 / (2 ** 2)

        # forward unitary FFT
        X = rfft2(x_3d)

        # extract hipass and lowpass components
        Xhi, Xlow = extractSemiFinestComponents(X, M1, M2, ctx.hipass, ctx.lowpass)

        # inverse unitary FFT
        imHi = irfft2(Xhi)
        imLow = irfft2(Xlow)

        # reshape components from 3D to 4D then output
        output_imHi = imHi.view(ctx.shape[0], ctx.shape[1], imHi.shape[1], imHi.shape[2])
        output_imLow = imLow.view(ctx.shape[0], ctx.shape[1], imLow.shape[1], imLow.shape[2])
        return output_imHi, output_imLow


    @staticmethod
    def backward(ctx, output_imHi, output_imLow):
        # reshape components from 4D to 3D
        depth = ctx.shape[0] * ctx.shape[1]
        imHi = torch.reshape(output_imHi, [depth, output_imHi.shape[2], output_imHi.shape[3]])
        imLow = torch.reshape(output_imLow, [depth, output_imLow.shape[2], output_imLow.shape[3]])

        # forward unitary FFT
        Xhi = rfft2(imHi)
        Xlow = rfft2(imLow)

        # zero pad lowpass component to the same size as hipass component
        Xlow_pad = padLowpassComponent(Xlow, ctx.shape[2], ctx.shape[3])

        # combine hipass and lowpass components
        X = Xhi + Xlow_pad

        # inverse unitary FFT
        x_rec = irfft2(X)

        # reshape reconstruction from 3D to 4D
        output_x = torch.reshape(x_rec, ctx.shape)
        return output_x, None, None, None, None





class coarseUpsampling(Function):
    """
    Implement forward backward upsampling from coarse scale to semi-finest scale.
    Inputs:
        x (torch.Tensor): input image of size (N, C, H, W)
        S1, S2: size of padded image
    Returns:
        torch.Tensor: xUp
    """

    @staticmethod
    def forward(ctx, x, S1, S2):
        # save for backward
        ctx.shape = x.shape
        ctx.N1 = ctx.shape[2]
        ctx.N2 = ctx.shape[3]
        ctx.S1 = S1
        ctx.S2 = S2

        # reshape input from 4D to 3D
        depth = ctx.shape[0] * ctx.shape[1]
        x_3d = torch.reshape(x, [depth, ctx.N1, ctx.N2])

        # forward unitary FFT
        X = rfft2(x_3d)

        # zero pad lowpass component to the same size as hipass component
        X_pad = padLowpassComponent(X, S1, S2)

        # inverse unitary FFT
        x_up = irfft2(X_pad)

        # reshape reconstruction from 3D to 4D
        xUp = x_up.view(ctx.shape[0], ctx.shape[1], ctx.S1, ctx.S2)
        return xUp


    @staticmethod
    def backward(ctx, xUp):
        # reshape input from 4D to 3D
        depth = ctx.shape[0] * ctx.shape[1]
        xUp_3d = torch.reshape(xUp, [depth, ctx.S1, ctx.S2])

        # get start and end index
        start_index = math.floor((ctx.S1 - ctx.N1) / 2)
        end_index = start_index + ctx.N1

        # forward unitary FFT
        X = rfft2(xUp_3d)

        # down sampling
        X_down = X[:, start_index:end_index, start_index:end_index, :]

        # inverse FFT
        x_down = irfft2(X_down)

        # reshape reconstruction from 3D to 4D
        xDown = x_down.view(ctx.shape[0], ctx.shape[1], ctx.N1, ctx.N2)
        return xDown, None, None





class semifinestUpsampling(Function):
    """
    Implement forward backward upsampling from semi-finest scale to finest scale.
    Inputs:
        (torch.Tensor) x: input image of size (N, C, H, W)
        S1, S2: size of padded image
    Returns:
        torch.Tensor: xUp
    """

    @staticmethod
    def forward(ctx, x, S1, S2):
        # save for backward
        ctx.shape = x.shape
        ctx.N1 = ctx.shape[2]
        ctx.N2 = ctx.shape[3]
        ctx.S1 = S1
        ctx.S2 = S2

        # reshape input from 4D to 3D
        depth = ctx.shape[0] * ctx.shape[1]
        x_3d = torch.reshape(x, [depth, ctx.N1, ctx.N2])

        # compute bigN1 and bigN2 for padding used
        M1 = ctx.S1 / 3
        M2 = ctx.S2 / 3
        bigN1 = 2 * math.floor(2 * M1) + 1
        bigN2 = 2 * math.floor(2 * M2) + 1

        # forward unitary FFT
        X = rfft2(x_3d)

        # zero pad lowpass component to the same size as hipass component
        X_pad = padLowpassComponent(X, bigN1, bigN2)

        # fold onto original image size
        X_up = wrapping(X_pad, S1, S2)

        # inverse unitary FFT
        x_up = irfft2(X_up)

        # reshape reconstruction from 3D to 4D
        xUp = x_up.view(ctx.shape[0], ctx.shape[1], ctx.S1, ctx.S2)
        return xUp


    @staticmethod
    def backward(ctx, xUp):
        # reshape input from 4D to 3D
        depth = ctx.shape[0] * ctx.shape[1]
        xUp_3d = torch.reshape(xUp, [depth, ctx.S1, ctx.S2])

        # Initialization: smooth periodic extension of high frequencies
        M1 = ctx.S1 / 3
        M2 = ctx.S2 / 3
        bigN1 = 2 * math.floor(2 * M1) + 1
        bigN2 = 2 * math.floor(2 * M2) + 1
        equiv_index_1 = 1 + np.mod(math.floor(ctx.S1 / 2) - math.floor(2 * M1) + np.arange(1, bigN1 + 1) - 1, ctx.S1)
        equiv_index_2 = 1 + np.mod(math.floor(ctx.S2 / 2) - math.floor(2 * M1) + np.arange(1, bigN2 + 1) - 1, ctx.S2)

        # change type and index starting at [0]
        equiv_index_1 = equiv_index_1.astype(int) - 1
        equiv_index_2 = equiv_index_2.astype(int) - 1

        # update M1, M2 for later used
        M1 = M1 / 2
        M2 = M2 / 2

        # matlab Xlow_index_1 is [18,19,...,50] but here [17,18,...,49]
        X_index_1 = np.arange(-math.floor(2.0 * M1), math.floor(2.0 * M1) + 1) + math.floor(4.0 * M1)
        X_index_2 = np.arange(-math.floor(2.0 * M2), math.floor(2.0 * M2) + 1) + math.floor(4.0 * M2)
        # change data type, -1 is not necessary here: because we pretend miss +1 at the above lines
        X_index_1 = X_index_1.astype(int)
        X_index_2 = X_index_2.astype(int)

        # forward unitary FFT
        X = rfft2(xUp_3d)

        # unfold image onto bigN1 x bigN2
        X = X[:, :, equiv_index_2, :][:, equiv_index_1, :, :]

        # down sample via extraction
        Xdown = X[:, :, X_index_2, :][:, X_index_1, :, :]

        # inverse unitary FFT
        x_down = irfft2(Xdown)

        # reshape reconstruction from 3D to 4D
        xDown = x_down.view(ctx.shape[0], ctx.shape[1], ctx.N1, ctx.N2)
        return xDown, None, None




