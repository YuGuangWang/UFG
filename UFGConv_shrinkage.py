import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import get_laplacian
import math
import argparse
import os.path as osp
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)


def rigrsure(x, N1, N2, col_idx):
    """
    Adaptive threshold selection using principle of Stein's Unbiased Risk Estimate (SURE).

    :param x: one block of wavelet coefficients, shape [num_nodes, num_hid_features] torch dense tensor
    :param N1: torch dense tensor with shape [num_nodes, num_hid_features]
    :param N2: torch dense tensor with shape [num_nodes, num_hid_features]
    :param col_idx: torch dense tensor with shape [num_hid_features]
    :return: thresholds stored in a torch dense tensor with shape [num_hid_features]
    """
    n, m = x.shape

    sx, _ = torch.sort(torch.abs(x), dim=0)
    sx2 = sx ** 2
    CS1 = torch.cumsum(sx2, dim=0)
    risks = (N1 + CS1 + N2 * sx2) / n
    best = torch.argmin(risks, dim=0)
    thr = sx[best, col_idx]

    return thr


def multiScales(x, r, Lev, num_nodes):
    """
    calculate the scales of the high frequency wavelet coefficients, which will be used for wavelet shrinkage.

    :param x: all the blocks of wavelet coefficients, shape [r * Lev * num_nodes, num_hid_features] torch dense tensor
    :param r: an integer
    :param Lev: an integer
    :param num_nodes: an integer which denotes the number of nodes in the graph
    :return: scales stored in a torch dense tensor with shape [(r - 1) * Lev] for wavelet shrinkage
    """
    for block_idx in range(Lev, r * Lev):
        if block_idx == Lev:
            specEnergy_temp = torch.unsqueeze(torch.sum(x[block_idx * num_nodes:(block_idx + 1) * num_nodes, :] ** 2), dim=0)
            specEnergy = torch.unsqueeze(torch.tensor(1.0), dim=0).to(x.device)
        else:
            specEnergy = torch.cat((specEnergy,
                                    torch.unsqueeze(torch.sum(x[block_idx * num_nodes:(block_idx + 1) * num_nodes, :] ** 2), dim=0) / specEnergy_temp))

    assert specEnergy.shape[0] == (r - 1) * Lev, 'something wrong in multiScales'
    return specEnergy


def simpleLambda(x, scale, sigma=1.0):
    """
    De-noising by Soft-thresholding. Author: David L. Donoho

    :param x: one block of wavelet coefficients, shape [num_nodes, num_hid_features] torch dense tensor
    :param scale: the scale of the specific input block of wavelet coefficients, a zero-dimensional torch tensor
    :param sigma: a scalar constant, which denotes the standard deviation of the noise
    :return: thresholds stored in a torch dense tensor with shape [num_hid_features]
    """
    n, m = x.shape
    thr = (math.sqrt(2 * math.log(n)) / math.sqrt(n) * sigma) * torch.unsqueeze(scale, dim=0).repeat(m)

    return thr


def waveletShrinkage(x, thr, mode='soft'):
    """
    Perform soft or hard thresholding. The shrinkage is only applied to high frequency blocks.

    :param x: one block of wavelet coefficients, shape [num_nodes, num_hid_features] torch dense tensor
    :param thr: thresholds stored in a torch dense tensor with shape [num_hid_features]
    :param mode: 'soft' or 'hard'. Default: 'soft'
    :return: one block of wavelet coefficients after shrinkage. The shape will not be changed
    """
    assert mode in ('soft', 'hard'), 'shrinkage type is invalid'

    if mode == 'soft':
        x = torch.mul(torch.sign(x), (((torch.abs(x) - thr) + torch.abs(torch.abs(x) - thr)) / 2))
    else:
        x = torch.mul(x, (torch.abs(x) > thr))

    return x


# function for pre-processing
@torch.no_grad()
def scipy_to_torch_sparse(A):
    A = sparse.coo_matrix(A)
    row = torch.tensor(A.row)
    col = torch.tensor(A.col)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data)

    return torch.sparse_coo_tensor(index, value, A.shape)


# function for pre-processing
def ChebyshevApprox(f, n):  # assuming f : [0, pi] -> R
    quad_points = 500
    c = np.zeros(n)
    a = np.pi / 2
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(a * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)

    return c


# function for pre-processing
def get_operator(L, DFilters, n, s, J, Lev):
    r = len(DFilters)
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)
    a = np.pi / 2  # consider the domain of masks as [0, pi]
    # Fast Tight Frame Decomposition (FTFD)
    FD1 = sparse.identity(L.shape[0])
    d = dict()
    for l in range(1, Lev + 1):
        for j in range(r):
            T0F = FD1
            T1F = ((s ** (-J + l - 1) / a) * L) @ T0F - T0F
            d[j, l - 1] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J + l - 1)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l - 1] += c[j][k] * TkF
        FD1 = d[0, l - 1]

    return d


class UFGConv(nn.Module):
    def __init__(self, in_features, out_features, r, Lev, num_nodes, shrinkage, sigma, bias=True):
        super(UFGConv, self).__init__()
        self.r = r
        self.Lev = Lev
        self.num_nodes = num_nodes
        self.shrinkage = shrinkage
        self.sigma = sigma
        self.crop_len = (Lev - 1) * num_nodes
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features).cuda())
            self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1).cuda())
            # self.N1 = torch.Tensor(list(num_nodes - 2 * np.arange(1, num_nodes + 1))).view(-1, 1).repeat(1, out_features).cuda()
            # self.N2 = torch.Tensor(list(np.arange(num_nodes))[::-1]).view(-1, 1).repeat(1, out_features).cuda()
            # self.col_idx = torch.tensor(list(range(out_features))).cuda()
        else:
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
            self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1))
            # self.N1 = torch.Tensor(list(num_nodes - 2 * np.arange(1, num_nodes + 1))).view(-1, 1).repeat(1, out_features)
            # self.N2 = torch.Tensor(list(np.arange(num_nodes))[::-1]).view(-1, 1).repeat(1, out_features)
            # self.col_idx = torch.tensor(list(range(out_features)))
        if bias:
            if torch.cuda.is_available():
                self.bias = nn.Parameter(torch.Tensor(out_features).cuda())
            else:
                self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.filter, 0.9, 1.1)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, d_list):
        # d_list is a list of matrix operators (torch sparse format), row-by-row
        # x is a torch dense tensor
        x = torch.matmul(x, self.weight)

        # Fast Tight Frame Decomposition
        x = torch.sparse.mm(torch.cat(d_list, dim=0), x)
        # the output x has shape [r * Lev * num_nodes, #Features]

        # Hadamard product in spectral domain
        x = self.filter * x
        # filter has shape [r * Lev * num_nodes, 1]
        # the output x has shape [r * Lev * num_nodes, #Features]

        # calculate the scales for thresholding
        ms = multiScales(x, self.r, self.Lev, self.num_nodes)

        # perform wavelet shrinkage
        for block_idx in range(self.Lev - 1, self.r * self.Lev):
            ms_idx = 0
            if block_idx == self.Lev - 1:  # low frequency block
                x_shrink = x[block_idx * self.num_nodes:(block_idx + 1) * self.num_nodes, :]
            else:  # remaining high frequency blocks with wavelet shrinkage
                x_shrink = torch.cat((x_shrink,
                                      waveletShrinkage(x[block_idx * self.num_nodes:(block_idx + 1) * self.num_nodes, :],
                                                       simpleLambda(x[block_idx * self.num_nodes:(block_idx + 1) * self.num_nodes, :],
                                                                    ms[ms_idx], self.sigma), mode=self.shrinkage)), dim=0)
                ms_idx += 1

        # Fast Tight Frame Reconstruction
        x_shrink = torch.sparse.mm(torch.cat(d_list[self.Lev - 1:], dim=1), x_shrink)

        if self.bias is not None:
            x_shrink += self.bias
        return x_shrink


class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, r, Lev, num_nodes, shrinkage='soft', sigma=1.0, dropout_prob=0.5):
        super(Net, self).__init__()
        self.GConv1 = UFGConv(num_features, nhid, r, Lev, num_nodes, shrinkage=shrinkage, sigma=sigma)
        self.GConv2 = UFGConv(nhid, num_classes, r, Lev, num_nodes, shrinkage=shrinkage, sigma=sigma)
        self.drop1 = nn.Dropout(dropout_prob)

    def forward(self, data, d_list):
        x = data.x  # x has shape [num_nodes, num_input_features]

        x = self.GConv1(x, d_list)
        x = self.drop1(x)
        x = self.GConv2(x, d_list)

        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='name of dataset (default: Cora)')
    parser.add_argument('--reps', type=int, default=10,
                        help='number of repetitions (default: 10)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--wd', type=float, default=0.01,
                        help='weight decay (default: 5e-3)')
    parser.add_argument('--nhid', type=int, default=16,
                        help='number of hidden units (default: 16)')
    parser.add_argument('--Lev', type=int, default=2,
                        help='level of transform (default: 2)')
    parser.add_argument('--s', type=float, default=2.5,
                        help='dilation scale > 1 (default: 2)')
    parser.add_argument('--n', type=int, default=2,
                        help='n - 1 = Degree of Chebyshev Polynomial Approximation (default: n = 2)')
    parser.add_argument('--FrameType', type=str, default='Haar',
                        help='frame type (default: Haar)')
    parser.add_argument('--dropout', type=float, default=0.7,
                        help='dropout probability (default: 0.5)')
    parser.add_argument('--shrinkage', type=str, default='soft',
                        help='soft or hard thresholding (default: soft)')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='standard deviation of the noise (default: 1.0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--filename', type=str, default='results',
                        help='filename to store results and the model (default: results)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Training on CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # load dataset
    dataname = args.dataset
    rootname = osp.join(osp.abspath(''), 'data', dataname)
    dataset = Planetoid(root=rootname, name=dataname)

    num_nodes = dataset[0].x.shape[0]
    L = get_laplacian(dataset[0].edge_index, num_nodes=num_nodes, normalization='sym')
    L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))

    lobpcg_init = np.random.rand(num_nodes, 1)
    lambda_max, _ = lobpcg(L, lobpcg_init)
    lambda_max = lambda_max[0]

    # extract decomposition/reconstruction Masks
    FrameType = args.FrameType

    if FrameType == 'Haar':
        D1 = lambda x: np.cos(x / 2)
        D2 = lambda x: np.sin(x / 2)
        DFilters = [D1, D2]
        RFilters = [D1, D2]
    elif FrameType == 'Linear':
        D1 = lambda x: np.square(np.cos(x / 2))
        D2 = lambda x: np.sin(x) / np.sqrt(2)
        D3 = lambda x: np.square(np.sin(x / 2))
        DFilters = [D1, D2, D3]
        RFilters = [D1, D2, D3]
    elif FrameType == 'Quadratic':  # not accurate so far
        D1 = lambda x: np.cos(x / 2) ** 3
        D2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
        D3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
        D4 = lambda x: np.sin(x / 2) ** 3
        DFilters = [D1, D2, D3, D4]
        RFilters = [D1, D2, D3, D4]
    else:
        raise Exception('Invalid FrameType')

    Lev = args.Lev  # level of transform
    s = args.s  # dilation scale
    n = args.n  # n - 1 = Degree of Chebyshev Polynomial Approximation
    J = np.log(lambda_max / np.pi) / np.log(s) + Lev - 1  # dilation level to start the decomposition
    r = len(DFilters)

    # get matrix operators
    d = get_operator(L, DFilters, n, s, J, Lev)
    # enhance sparseness of the matrix operators (optional)
    # d[np.abs(d) < 0.001] = 0.0
    # store the matrix operators (torch sparse format) into a list: row-by-row
    d_list = list()
    for i in range(r):
        for l in range(Lev):
            d_list.append(scipy_to_torch_sparse(d[i, l]).to(device))

    '''
    Training Scheme
    '''

    # Hyper-parameter Settings
    learning_rate = args.lr
    weight_decay = args.wd
    nhid = args.nhid

    # extract the data
    data = dataset[0].to(device)

    # create result matrices
    num_epochs = args.epochs
    num_reps = args.reps
    epoch_loss = dict()
    epoch_acc = dict()
    epoch_loss['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['test_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['test_mask'] = np.zeros((num_reps, num_epochs))
    saved_model_val_acc = np.zeros(num_reps)
    saved_model_test_acc = np.zeros(num_reps)

    for rep in range(num_reps):
        print('****** Rep {}: training start ******'.format(rep + 1))
        max_acc = 0.0

        # initialize the model
        model = Net(dataset.num_node_features, nhid, dataset.num_classes, r, Lev, num_nodes,
                    shrinkage=args.shrinkage, sigma=args.sigma, dropout_prob=args.dropout).to(device)

        # initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # initialize the learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        # training
        for epoch in range(num_epochs):
            # training mode
            model.train()
            optimizer.zero_grad()
            out = model(data, d_list)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # evaluation mode
            model.eval()
            out = model(data, d_list)
            for i, mask in data('train_mask', 'val_mask', 'test_mask'):
                pred = out[mask].max(dim=1)[1]
                correct = float(pred.eq(data.y[mask]).sum().item())
                e_acc = correct / mask.sum().item()
                epoch_acc[i][rep, epoch] = e_acc

                e_loss = F.nll_loss(out[mask], data.y[mask])
                epoch_loss[i][rep, epoch] = e_loss

            # scheduler.step(epoch_loss['val_mask'][rep, epoch])

            # print out results
            print('Epoch: {:3d}'.format(epoch + 1),
                  'train_loss: {:.4f}'.format(epoch_loss['train_mask'][rep, epoch]),
                  'train_acc: {:.4f}'.format(epoch_acc['train_mask'][rep, epoch]),
                  'val_loss: {:.4f}'.format(epoch_loss['val_mask'][rep, epoch]),
                  'val_acc: {:.4f}'.format(epoch_acc['val_mask'][rep, epoch]),
                  'test_loss: {:.4f}'.format(epoch_loss['test_mask'][rep, epoch]),
                  'test_acc: {:.4f}'.format(epoch_acc['test_mask'][rep, epoch]))

            # save model
            if epoch_acc['val_mask'][rep, epoch] > max_acc:
                torch.save(model.state_dict(), args.filename + '.pth')
                print('=== Model saved at epoch: {:3d}'.format(epoch + 1))
                max_acc = epoch_acc['val_mask'][rep, epoch]
                record_test_acc = epoch_acc['test_mask'][rep, epoch]

        saved_model_val_acc[rep] = max_acc
        saved_model_test_acc[rep] = record_test_acc
        print('#### Rep {0:2d} Finished! val acc: {1:.4f}, test acc: {2:.4f} ####\n'.format(rep + 1, max_acc, record_test_acc))

    print('***************************************************************************************************************************')
    print('Average test accuracy over {0:2d} reps: {1:.4f} with stdev {2:.4f}'.format(num_reps, np.mean(saved_model_test_acc), np.std(saved_model_test_acc)))
    print('dataset:', args.dataset, '; epochs:', args.epochs, '; reps:', args.reps, '; learning_rate:', args.lr, '; weight_decay:', args.wd, '; nhid:', args.nhid,
          '; Lev:', args.Lev)
    print('s:', args.s, '; n:', args.n, '; FrameType:', args.FrameType, '; dropout:', args.dropout, '; seed:', args.seed, '; filename:', args.filename)
    print('shrinkage:', args.shrinkage, '; sigma:', args.sigma)
    print('\n')
    print(args.filename + '.pth', 'contains the saved model and ', args.filename + '.npz', 'contains all the values of loss and accuracy.')
    print('***************************************************************************************************************************')

    # save the results
    np.savez(args.filename + '.npz',
             epoch_train_loss=epoch_loss['train_mask'],
             epoch_train_acc=epoch_acc['train_mask'],
             epoch_valid_loss=epoch_loss['val_mask'],
             epoch_valid_acc=epoch_acc['val_mask'],
             epoch_test_loss=epoch_loss['test_mask'],
             epoch_test_acc=epoch_acc['test_mask'],
             saved_model_val_acc=saved_model_val_acc,
             saved_model_test_acc=saved_model_test_acc)