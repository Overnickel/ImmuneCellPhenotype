from __future__ import print_function
import argparse
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

class ImmuneDataset(Dataset):

    def __init__(self, tsv_file, transform=None):
        self.data = pd.read_csv(tsv_file, sep='\t', index_col=0) 
        self.transform = transform
        self.phenotype = {'desert': 0, 'excluded': 1, 'inflamed': 2}

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        sample = self.data.iloc[:-1, idx].values.astype(np.float64), self.phenotype[self.data.iloc[-1, idx]]
        sample = torch.from_numpy(sample[0]).type(torch.FloatTensor), torch.LongTensor([sample[1]])
        if self.transform:
            sample = self.transform(sample)
        return sample

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(224, 32)
        self.fc2 = nn.Linear(32, 3)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target.squeeze(1))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.sampler),
                100. * (batch_idx + 1) / len(train_loader), loss.item()))

def test(args, model, device, test_loader): 
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target.squeeze(1)).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.sampler),
        100. * correct / len(test_loader.sampler)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Immune Phenotype Classification')
    parser.add_argument('--input-path', default='./immunephenotype.tsv',
                        help='input dataset path (default: ./immunephenotype.tsv)')
    parser.add_argument('--num-workers', type=int, default=0, metavar='N',
                        help='how many subprocesses to use for data loading (default: 0')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size (default: 4)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    # parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    #                     help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight-decay', type=float, default=0.001, metavar='WD',
                        help='weight decay (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--split-ratio', type=float, default=0.8, metavar='SR',
                        help='the ratio: trainset / totalset (default: 0.8)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # train/test sets split
    immunedataset = ImmuneDataset(args.input_path)
    dataset_size = len(immunedataset)
    indices = range(dataset_size)
    split = int(np.floor(args.split_ratio * dataset_size))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    kwargs = {'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(immunedataset, batch_size=args.batch_size, 
                                               sampler=train_sampler, num_workers=args.num_workers, timeout=1, **kwargs)
    test_loader = torch.utils.data.DataLoader(immunedataset, batch_size=args.batch_size,
                                              sampler=test_sampler, num_workers=args.num_workers, timeout=1, **kwargs)


    model = Net().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
