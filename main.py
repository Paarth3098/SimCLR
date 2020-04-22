import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

from PIL import Image
from torchvision import transforms
from torchvision import datasets

from model import Model


# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in tqdm(data_loader):
        pos_1, pos_2 = pos_1.to(device), pos_2.to(device)
        # print('pos_1 :', pos_1.shape[0])
        # print('pos_2 :', pos_2.shape)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # print('out_1 :', out_1.shape)
        # print('out_2 :', out_2.shape)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # print('out :', out.shape)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        # print(torch.ones_like(sim_matrix).shape)
        # print(torch.eye(2*batch_size).shape)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_1.shape[0], -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank, target_vec = 0.0, 0.0, 0, [], []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.to(device))
            feature_bank.append(feature)
            target_vec.append(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        target_vec = torch.cat(target_vec, dim=0).t().contiguous()
        # [N]
        # classes = memory_data_loader.dataset.classes
        # map = {'pouring':0, 'not_pouring':1}
        feature_labels = torch.tensor(target_vec, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.to(device), target.to(device)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


class SimCLR_Dataset(torch.utils.data.Dataset):
    # print('Inside wrapper')
    '''Dataset Wrapper for SimCLR'''

    def __init__(self, data, transform, root, train=True):
        self.data = data
        self.transform = transform
        self.train = train
        self.root = root
        self.classes = self._get_classes(self.root)

    def __getitem__(self, index):
        img, target = self.data[index]

        pos_1 = self.transform(img)
        pos_2 = self.transform(img)

        return pos_1, pos_2, target

    def __len__(self):
        return len(self.data)

    def _get_classes(self, dir):
        classes = [d.name for d in os.scandir(dir)]
        return classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--dir_path', default='/sandbox_classification_data', type=str, help='Data directory')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=100, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--resume', '-r', type=str, default='', help='Checkpoint path for resume / test.')

    # args parser
    args = parser.parse_args()
    dir = args.dir_path
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # data prepare

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

    data_dir = dir
    traindir = os.path.join(data_dir, 'train/')
    valdir = os.path.join('/media/neo/krypton/data/data_ptc/pouring_classification_data/ptc_dataset/', 'val/')

    train_dataset = datasets.ImageFolder(traindir)
    train_dataset = SimCLR_Dataset(train_dataset, transform=train_transform, root=traindir, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True
    )
    mem_dataset = datasets.ImageFolder(traindir)
    mem_dataset = SimCLR_Dataset(mem_dataset, transform=test_transform, root=traindir, train=False)
    memory_loader = DataLoader(
        mem_dataset,
        batch_size=int(batch_size*0.5),
        num_workers=16,
        shuffle = False,
        pin_memory=True
    )
    test_dataset = datasets.ImageFolder(valdir)
    test_dataset = SimCLR_Dataset(test_dataset,
                                  transform=test_transform,
                                  root=valdir,
                                  train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(batch_size*0.5),
        num_workers=16,
        shuffle = False,
        pin_memory=True
    )

    # # memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
    # memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    # test_data = utils.DataPair(root=data_dir, train =False, transfrom=utils.test_transform)
    # # test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
    # # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim).to(device)
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).to(device),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    c = len(mem_dataset.classes)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Model restored from epoch:', start_epoch)
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        scheduler.step()
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/iter_1/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }
        model.save(checkpoint, 'results/iter_1/{}_epoch_{}_checkpoint.pth'.format(save_name_pre,epoch))
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'results/iter_1/{}_model.pth'.format(save_name_pre))
