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
from util import LoadData, Initializer, EarlyStopping, Loss_l2_Inv, Loss_Mix_Inv, double_conv, Loss_l2_All
from util import getFirstLowpass, getHipassLowpass
from ResCoronaeNet_layer import finestCoronaeDecLayer, semiFinestCoronaeDecLayer, coarseUpsampling, semifinestUpsampling
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
patience = 20
scheduler = 'CosLR'
loss_weight = [2, 2, 1]

# data path
path = 'data/Curvelet_Inv/Ellipses'

# generate data loaders
train_loader = LoadData(path_datasets=path, mode='train', minibatch_size=batch_size)
valid_loader = LoadData(path_datasets=path, mode='val', minibatch_size=batch_size)

# choose loss function and define weight
loss_type = 'Weighted_l2_All'



# construct summary writer
train_writer = SummaryWriter(comment = os.path.join(path, 'train'))
valid_writer = SummaryWriter(comment = os.path.join(path, 'val'))






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
class ResCoronaeNet(nn.Module):
    """
    Construct Fast Discrete Curvelet Network (3 scales with 32 wedges in semi-finest scale)
    """

    def __init__(self):
        super(ResCoronaeNet, self).__init__()
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
        xsf_cat = torch.cat((xsf, xf_low), dim=1)
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
        xsf_ = torch.cat((xsf_hi, xc_up), dim=1)
        xsf_ = self.conv_sf3(xsf_)
        xsf_ = self.conv_sf4(xsf_)
        xsf_up = self.semifinestUp(xsf_)

        # finest scale
        xf_ = torch.cat((xf_hi, xsf_up), dim=1)
        xf_ = self.conv_f2(xf_)

        # output with residual
        xF = self.convF(xf_)+finestIm
        xSF = self.convSF(xf_)+self.semifinestUp(semiIm)
        xC = self.convC(xf_)+self.semifinestUp(self.coarseUp(coarseIm))

        xf_output = torch.cat((xF, xSF, xC), dim=1)
        return xf_output




### initialize network and specify optimizer ###

# initialize the FDCNet
model = ResCoronaeNet().to(device)
print('train ResCoronaeNet')

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total_params)

# Xavier initialization
Initializer.initialize(model=model, initialization=init.xavier_uniform_, gain=init.calculate_gain('relu'))

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=learningRate)

# select scheduler for learning
if scheduler == 'CosLR':
    T_max = n_epochs * len(train_loader) + 1
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
elif scheduler == 'StepLR':
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
else:
    print('no scheduler')





### train the model ###

def train_model(model, patience, n_epochs):

    # to track the training & validation loss as the model trains
    train_losses = []
    valid_losses = []

    # to track the average training & validation loss per epoch as the model trains
    avg_train_losses = []
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=os.path.join(path, 'checkpoint.pth'))

    print("start training")
    for epoch in range(1, n_epochs + 1):

        # train the model
        model.train() # prepare model for training

        # train model according to batch size
        for train_imF, train_imSF, train_imC, train_label in train_loader:

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output_label = model(train_imF.to(device), train_imSF.to(device), train_imC.to(device))
            loss = criterion(output_label, train_label.to(device))
            loss.backward()

            # perform a single optimization step
            optimizer.step()

            scheduler.step()

            # record training loss
            train_losses.append(loss.item())
        '''
        for name, parms in model.named_parameters():
            print('-->name:', name, '-->grad_requires:', parms.requires_grad, '-->grad_value:', parms.grad)'''

        # validate the model
        model.eval() # prepare model for validation

        # validate model on validation set
        for val_imF, val_imSF, val_imC, val_label in valid_loader:

            # forward pass
            val_output = model(val_imF.to(device), val_imSF.to(device), val_imC.to(device))

            # calculate the loss
            loss = criterion(val_output, val_label.to(device))

            # record validation loss
            valid_losses.append(loss.item())


        # print training and validation metrics
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        # record losses and print
        train_writer.add_scalar("Loss/train", train_loss, epoch)
        valid_writer.add_scalar("Loss/valid", valid_loss, epoch)

        epoch_len = len(str(n_epochs))
        print("[ {:>{}}/{:>{}} ] train_loss: {} valid_loss: {}".format(epoch, epoch_len, n_epochs, epoch_len, train_loss, valid_loss))

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []


        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
    return model, avg_train_losses, avg_valid_losses





### start training model ###

# train model
model, train_loss, valid_loss = train_model(model, patience, n_epochs)




### visualize checkpoint ###

fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1),valid_loss, label='Validation Loss')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss))+1
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 0.01) # consistent scale
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('loss_plot.png', bbox_inches='tight')

