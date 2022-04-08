'''
Written 2021 by Bolin Pan, UCL
'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import os
import matplotlib.pyplot as plt
import time
import scipy.io as io

from torch import optim
from util import LoadData, Initializer, EarlyStopping, Loss_l2_Inv, double_conv, Loss_Mix_Inv, Loss_l2_All
from util import getFirstLowpass, getHipassLowpass
from CorNet_layer import finestCoronaeDecLayer, semiFinestCoronaeDecLayer, coarseUpsampling, semifinestUpsampling
from tensorboardX import SummaryWriter



# Display pytorch version.
print(torch.__version__)

np.random.seed(42)
torch.manual_seed(42)

# select GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"








### Initialization ###

# original image size
N1 = 192
N2 = 192

# parameters for network
device = "cuda"
learningRate = 1e-3
batch_size = 2
n_epochs = 200
patience = 20 # 10 - ResUNet, UNet
scheduler = 'CosLR'
loss_weight = [2, 2, 1]

# data path
path = 'data/Curvelet_Inv/Ball'

# generate data loaders
test_loader = LoadData(path_datasets=path, mode='test', minibatch_size=batch_size)

# choose loss function and define weight
loss_type = 'Weighted_l2_Inv'

# construct summary writer
test_writer = SummaryWriter(comment = os.path.join(path,'test'))






### generate filters ###

# generate first lowpass
lowpass_first = getFirstLowpass(N1, N2, 32*batch_size)

# generate hipass and lowpass for finest scale decomposition
M1f = N1 / (3 * 2)
M2f = N2 / (3 * 2)
hipassF, lowpassF = getHipassLowpass(M1f, M2f, 32*batch_size)

# size for coarse scale upsampling
n1 = hipassF.shape[1]
n2 = hipassF.shape[2]

# generate hipass and lowpass for semi-finest scale decomposition
M1 = N1 / 3
M2 = N2 / 3
M1sf = M1 / (2 ** 2)
M2sf = M2 / (2 ** 2)
hipassSF, lowpassSF = getHipassLowpass(M1sf, M2sf, 64*batch_size)




### construct decomposition and upsampling layers ###

# coronae decomposition in finest scale
class finestDec(nn.Module):
    def __init__(self, lowpass_first = lowpass_first, hipass = hipassF, lowpass = lowpassF, device = device):
        super(finestDec, self).__init__()
        self.lowpass_first = lowpass_first.to(device)
        self.hipass = hipass.to(device)
        self.lowpass = lowpass.to(device)

    def forward(self, input):
        return finestCoronaeDecLayer.apply(input, self.lowpass_first, self.hipass, self.lowpass)


# coronae decomposition in semi-finest scale
class semifinestDec(nn.Module):
    def __init__(self, N1 = N1, N2 =  N2, hipass = hipassSF, lowpass = lowpassSF, device = device):
        super(semifinestDec, self).__init__()
        self.N1 = N1
        self.N2 = N2
        self.hipass = hipass.to(device)
        self.lowpass = lowpass.to(device)

    def forward(self, input):
        return semiFinestCoronaeDecLayer.apply(input, self.N1, self.N2, self.hipass, self.lowpass)


# coarse scale upsampling
class coarseUp(nn.Module):
    def __init__(self, S1 = n1, S2 = n1):
        super(coarseUp, self).__init__()
        self.S1 = S1
        self.S2 = S2

    def forward(self, input):
        return coarseUpsampling.apply(input, self.S1, self.S2)


# coarse scale upsampling
class semifinestUp(nn.Module):
    def __init__(self, S1 = N1, S2 = N2):
        super(semifinestUp, self).__init__()
        self.S1 = S1
        self.S2 = S2

    def forward(self, input):
        return semifinestUpsampling.apply(input, self.S1, self.S2)






## define loss function
if loss_type == 'Weighted_l2_Inv':
    criterion = Loss_l2_Inv(weights=loss_weight)
elif loss_type == 'Weighted_Mix_Inv':
    criterion = Loss_Mix_Inv(weights=loss_weight)
elif loss_type == 'Weighted_l2_All':
    criterion = Loss_l2_All(weights =loss_weight)
else:
    print('Loss not defined')




# construct network
class CurveNet(nn.Module):
    """
    Construct Fast Discrete Curvelet Network (3 scales with 32 wedges in semi-finest scale)
    """

    def __init__(self):
        super(CurveNet, self).__init__()
        # convolution along finest scale
        self.conv_f1 = double_conv(2, 32)
        self.conv_f2 = double_conv(64, 32)
        self.convF = nn.Conv2d(32, 2, 1)
        self.convSF = nn.Conv2d(32, 2, 1)
        self.convC = nn.Conv2d(32, 2, 1)


        # convolution along semi-finest scale
        self.conv_sf1 = double_conv(2, 32)
        self.conv_sf2 = double_conv(64, 64)
        self.conv_sf3 = double_conv(128, 64)
        self.conv_sf4 = double_conv(64, 32)

        # convolution along coarse scale
        self.conv_c1 = double_conv(2, 64)
        self.conv_c2 = double_conv(128, 128)
        self.conv_c3 = double_conv(128, 128)
        self.conv_c4 = double_conv(128, 64)

        # finest scale decomposition
        self.finestDec = finestDec()

        # semi-finest scale decomposition
        self.semifinestDec = semifinestDec()

        # coarse scale to semi-finest scale upsampling
        self.coarseUp = coarseUp()

        # semi-finest scale to finest scale upsampling
        self.semifinestUp = semifinestUp()


    def forward(self, finestIm, semiIm, coarseIm):
        ''' encoder '''
        # finest scale
        xf = self.conv_f1(finestIm)
        xf_hi, xf_low = self.finestDec(xf)

        # semi-finest scale
        xsf = self.conv_sf1(semiIm)
        xsf_cat = torch.cat((xsf, xf_low), dim = 1)
        xsf = self.conv_sf2(xsf_cat)
        xsf_hi, xsf_low = self.semifinestDec(xsf)

        # coarse scale
        xc = self.conv_c1(coarseIm)
        xc_cat = torch.cat((xc, xsf_low), dim=1)
        xc = self.conv_c2(xc_cat)
        xc = self.conv_c3(xc)
        xc = self.conv_c4(xc)
        xc_up = self.coarseUp(xc)


        ''' decoder '''
        # semi-finest scale
        xsf_ = torch.cat((xsf_hi, xc_up), dim = 1)
        xsf_ = self.conv_sf3(xsf_)
        xsf_ = self.conv_sf4(xsf_)
        xsf_up = self.semifinestUp(xsf_)

        # finest scale
        xf_ = torch.cat((xf_hi, xsf_up), dim = 1)
        xf_ = self.conv_f2(xf_)
        xF = self.convF(xf_)
        xSF = self.convSF(xf_)
        xC = self.convC(xf_)
        xf_output = torch.cat((xF, xSF, xC), dim=1)
        return xf_output



### initialize network and specify optimizer ###

# initialize the FDCNet
model = CurveNet().to(device)
print('using MSCurveNet')





### load trained model ###

# load the last checkpoint with the best model
model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

# prep model for evaluation
model.eval()






### test the trained network ###

# initialize
test_losses = []
path = os.path.join(path, 'eval/eval_')
eval = torch.tensor([]).to(device)
i = 0

for test_imF, test_imSF, test_imC, test_label in test_loader:

    # forward pass to compute the predicted outputs
    eval = model(test_imF.to(device), test_imSF.to(device), test_imC.to(device))

    # calculate the loss
    loss = criterion(eval, test_label.to(device))
    test_losses.append(loss.item())

    # save mat output
    i += 1
    savePath = path + str(i)
    io.savemat(savePath, {"eval": eval.cpu().detach().numpy()})

# print avg test loss
test_loss = np.average(test_losses)
print("Test Loss: {}".format(test_loss))






