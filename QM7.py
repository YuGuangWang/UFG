import torch
import torch.nn.functional as F
import scipy.io
from torch_geometric.data import InMemoryDataset, download_url, Data


class QM7(InMemoryDataset):
    url = 'http://quantum-machine.org/data/qm7.mat'

    def __init__(self, root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(QM7, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'qm7.mat'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data = scipy.io.loadmat(self.raw_paths[0])
        coulomb_matrix = torch.from_numpy(data['X'])
        target = torch.from_numpy(data['T']).to(torch.float).squeeze(0)
        mean = torch.mean(target).item()
        std = torch.sqrt(torch.var(target)).item()

        data_list = []
        for i in range(target.shape[0]):
            edge_index = coulomb_matrix[i].nonzero().t().contiguous()
            edge_attr = coulomb_matrix[i, edge_index[0], edge_index[1]]
            y_origin = target[i].item()
            y = (target[i].item() - mean) / std
            data = Data(edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.num_nodes = edge_index.max().item() + 1
            data.y_origin = y_origin
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def qm7_test(model, loader, device, mean, std):
    model.eval()
    loss = 0.
    for data in loader:
        data = data.to(device)
        out = model(data)
        out = out * std + mean
        loss += F.l1_loss(out, data.y_origin, reduction='sum').item()
    return loss / len(loader.dataset)


def qm7_test_train(model, loader, device):
    model.eval()
    loss = 0.
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss += F.mse_loss(out, data.y, reduction='sum').item()
    return loss / len(loader.dataset)