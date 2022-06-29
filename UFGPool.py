import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import get_laplacian
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import degree
import argparse
import os.path as osp
from QM7 import QM7, qm7_test, qm7_test_train
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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


# function for early_stopping
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
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
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def UFGPool(x, batch, batch_size, d_list, d_index, aggre_mode='sum'):
    """
    Using Undecimated Framelet Transform for graph pooling.

    :param x: batched hidden representation. shape: [# Node_Sum_Batch, # Hidden Units]
    :param batch: batch index. shape: [# Node_Sum_Batch]
    :param batch_size: integer batch size.
    :param d_list: a list of matrix operators, where each element is a torch sparse tensor stored in a list.
    :param d_index: a list of index tensors, where each element is a torch dense tensor used for aggregation.
    :param aggre_mode: aggregation mode. choices: sum, max, and avg. (default: sum)
    :return: batched vectorial representation for the graphs in the batch.
    """
    if aggre_mode == 'sum':
        f = global_add_pool
    elif aggre_mode == 'avg':
        f = global_mean_pool
    elif aggre_mode == 'max':
        f = global_max_pool
    else:
        raise Exception('aggregation mode is invalid')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(batch_size):
        # extract the i-th graph
        bi = (batch == i)
        coefs = torch.sparse.mm(scipy_to_torch_sparse(d_list[i][0]).to(device), x[bi, :])###shape 90 x 64
        idx = torch.tensor(d_index[i][0]).to(device)
        if i == 0:
            x_pool = f(coefs, idx).flatten()#####output should be r-1 * lev+1 * 64
        else:
            x_pool = torch.vstack((x_pool, f(coefs, idx).flatten())) ##d_index contain all batch

    return x_pool ##batch_size*64 vector

class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, r, Lev, dropout_prob=0.5):
        super(Net, self).__init__()
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes
        self.GConv1 = GCNConv(num_features, nhid)
        self.GConv2 = GCNConv(nhid, nhid)

        self.fc = nn.Sequential(nn.Linear(((r - 1) * Lev + 1) * nhid, nhid),
                                #nn.BatchNorm1d(nhid),
                                nn.ReLU(),
                                nn.Dropout(dropout_prob),
                                nn.Linear(nhid, num_classes),)
                                #nn.BatchNorm1d(num_classes))

    def forward(self, data):
        x, edge_index, batch, d, d_index = data.x, data.edge_index, data.batch, data.d, data.d_index
        batch_size = int(batch.max() + 1)

        # two convolutional layers
        x = F.relu(self.GConv1(x, edge_index))
        x = F.relu(self.GConv2(x, edge_index))

        # one global pooling layer
        x = UFGPool(x, batch, batch_size, d, d_index)
        x = self.fc(x)
        #x = F.log_softmax(x, dim=-1)
        if num_classes == 1:
            return x.view(-1)
        else: 
            return x


def MyDataset(dataset, Lev, s, n, FrameType='Haar', add_feature=False, QM7=False):
    if FrameType == 'Haar':
        D1 = lambda x: np.cos(x / 2)
        D2 = lambda x: np.sin(x / 2)
        DFilters = [D1, D2]
    elif FrameType == 'Linear':
        D1 = lambda x: np.square(np.cos(x / 2))
        D2 = lambda x: np.sin(x) / np.sqrt(2)
        D3 = lambda x: np.square(np.sin(x / 2))
        DFilters = [D1, D2, D3]
    elif FrameType == 'Quadratic':  # not accurate so far
        D1 = lambda x: np.cos(x / 2) ** 3
        D2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
        D3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
        D4 = lambda x: np.sin(x / 2) ** 3
        DFilters = [D1, D2, D3, D4]
    else:
        raise Exception('Invalid FrameType')
    r = len(DFilters)

    dataset1 = list()
    label=list()
    for i in range(len(dataset)):
        if add_feature:
            raise Exception('this function has not been completed')  # will add this function as required
        else:
            if QM7:
                x_qm7 = torch.ones(dataset[i].num_nodes, num_features)
                data1 = Data(x=x_qm7, edge_index=dataset[i].edge_index, y=dataset[i].y)
                data1.y_origin = dataset[i].y_origin
            else:
                data1 = Data(x=dataset[i].x, edge_index=dataset[i].edge_index, y=dataset[i].y)
                
        if QM7:
            label.append(dataset[i].y_origin)
        # get graph Laplacian
        num_nodes = data1.x.shape[0]
        L = get_laplacian(dataset[i].edge_index, num_nodes=num_nodes, normalization='sym')
        L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))
        # calculate lambda max
        lobpcg_init = np.random.rand(num_nodes, 1)
        lambda_max, _ = lobpcg(L, lobpcg_init)
        lambda_max = lambda_max[0]
        J = np.log(lambda_max / np.pi) / np.log(s) + Lev - 1  # dilation level to start the decomposition
        # get matrix operators
        d = get_operator(L, DFilters, n, s, J, Lev)
        for m in range(1, r):
            for q in range(Lev):
                if (m == 1) and (q == 0):
                    d_aggre = d[m, q]
                else:
                    d_aggre = sparse.vstack((d_aggre, d[m, q]))
        d_aggre = sparse.vstack((d[0, Lev - 1], d_aggre)) ###stack the n x n matrix
        data1.d = [d_aggre]
        # get d_index
        a = [i for i in range((r - 1) * Lev + 1)]##len=3
        data1.d_index=[[a[i // num_nodes] for i in range(len(a) * num_nodes)]]##3*num [0,1,2;,0,1,2...]
        # append data1 into dataset1
        dataset1.append(data1)    
    if QM7:
        mean = torch.mean(torch.Tensor(label)).item()
        std = torch.sqrt(torch.var(torch.Tensor(label))).item()
        return dataset1, r, mean, std    
    else:
        return dataset1, r


def test(model, loader, device):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(device)
        out = model(data)
        if len(out.shape)<2:
            out = out.unsqueeze(0)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.cross_entropy(out, data.y,reduction='sum').item()
    return correct / len(loader.dataset), loss / len(loader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PROTEINS',
                        help='name of dataset (default: PROTEINS), options: PROTEINS, Mutagenicity, D&D, NCI1')
    parser.add_argument('--reps', type=int, default=10,
                        help='number of repetitions (default: 10)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--patience', type=int, default=20,
                        help='early stopping criteria (default: 20)')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--wd', type=float, default=5e-3,
                        help='weight decay (default: 5e-3)')
    parser.add_argument('--nhid', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--batch_size', type=int, default=53,
                        help='batch size (default: 64)')
    parser.add_argument('--Lev', type=int, default=2,
                        help='level of transform (default: 2)')
    parser.add_argument('--s', type=float, default=2,
                        help='dilation scale > 1 (default: 2)')
    parser.add_argument('--n', type=int, default=2,
                        help='n - 1 = Degree of Chebyshev Polynomial Approximation (default: n = 2)')
    parser.add_argument('--FrameType', type=str, default='Haar',
                        help='frame type (default: Haar)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout probability (default: 0.5)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--filename', type=str, default='results',
                        help='filename to store results and the model (default: results)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Training on CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    dataname = args.dataset
    path = osp.join(osp.abspath(''), 'data', dataname)
    if dataname == 'qm7':
        dataset = QM7(path)
        num_features = 5
        num_classes = 1 
        loss_criteria = F.mse_loss
        dataset, rr, mean, std = MyDataset(dataset, args.Lev, args.s, args.n, FrameType=args.FrameType, QM7=True)
    else:
        dataset = TUDataset(path, name=dataname)
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        loss_criteria = F.cross_entropy
        if num_features == 0:
            dataset_temp = list()
            for i in range(len(dataset)):
                x = degree(dataset[i].edge_index[0], dataset[i].num_nodes).view(-1,1)
                data_i = Data(x=x,edge_index=dataset[i].edge_index,y=dataset[i].y)
                dataset_temp.append(data_i)
            dataset,rr = MyDataset(dataset_temp, args.Lev, args.s, args.n, FrameType=args.FrameType)
            num_features = 1
        else:
            dataset,rr = MyDataset(dataset, args.Lev, args.s, args.n, FrameType=args.FrameType)
            
    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    print("num dataset:",num_training,num_val,num_test)
    # Parameter Setting
    batch_size = args.batch_size
    learning_rate = args.lr
    weight_decay = args.wd
    nhid = args.nhid
    epochs = args.epochs
    num_reps = args.reps

    # create results matrix
    epoch_train_loss = np.zeros((num_reps, epochs))
    epoch_train_acc = np.zeros((num_reps, epochs))
    epoch_valid_loss = np.zeros((num_reps, epochs))
    epoch_valid_acc = np.zeros((num_reps, epochs))
    epoch_test_loss = np.zeros((num_reps, epochs))
    epoch_test_acc = np.zeros((num_reps, epochs))
    saved_model_loss = np.zeros(num_reps)
    saved_model_acc = np.zeros(num_reps)

    # training
    for r in range(num_reps):
        training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

        train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        model = Net(num_features, nhid, num_classes, rr, args.Lev, dropout_prob=args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, verbose=True)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=args.filename+'_latest.pth')

        # start training
        min_loss = 1e10
        patience = 0
        print("****** Rep {}: Training start ******".format(r+1))
        for epoch in range(epochs):
            model.train()
            for i, data in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data).squeeze(0)
                loss = loss_criteria(out, data.y, reduction='sum')
                loss.backward()
                optimizer.step()
            if dataname == 'qm7':
                train_loss = qm7_test_train(model, train_loader, device)
                val_loss = qm7_test(model, val_loader, device, mean, std)
                test_loss = qm7_test(model, test_loader, device, mean, std)
                print("Epoch {}: Training loss: {:5f}, Validation loss: {:5f}, Test loss: {:.5f}".format(epoch+1, train_loss, val_loss, test_loss))
            else:
                train_acc, train_loss = test(model, train_loader, device)
                val_acc, val_loss = test(model, val_loader, device)
                test_acc, test_loss = test(model, test_loader, device)
                epoch_train_acc[r, epoch],epoch_valid_acc[r, epoch],epoch_test_acc[r, epoch] = train_acc,val_acc,test_acc
                print("Epoch {}: Training accuracy: {:.5f}; Validation accuracy: {:.5f}; Test accuracy: {:.5f}".format(epoch+1, train_acc, val_acc, test_acc))
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping \n")
                break
            scheduler.step(val_loss)
            
            epoch_train_loss[r, epoch] = train_loss            
            epoch_valid_loss[r, epoch] = val_loss            
            epoch_test_loss[r, epoch] = test_loss

        # Test
        print("****** Test start ******")
        model = Net(num_features, nhid, num_classes, rr, args.Lev, dropout_prob=args.dropout).to(device)
        model.load_state_dict(torch.load(args.filename+'_latest.pth'))
        if dataname == 'qm7':
            test_loss = qm7_test(model, test_loader, device, mean, std)
            print("Test Loss: {:.5f}".format(test_loss))
        else:
            test_acc, test_loss = test(model, test_loader, device)
            saved_model_acc[r] = test_acc
            print("Test accuracy: {:.5f}".format(test_acc))
        saved_model_loss[r] = test_loss

    # save the results
    np.savez(args.filename + '.npz',
             epoch_train_loss=epoch_train_loss,
             epoch_train_acc=epoch_train_acc,
             epoch_valid_loss=epoch_valid_loss,
             epoch_valid_acc=epoch_valid_acc,
             epoch_test_loss=epoch_test_loss,
             epoch_test_acc=epoch_test_acc,
             saved_model_loss=saved_model_loss,
             saved_model_acc=saved_model_acc)
