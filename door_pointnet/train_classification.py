from __future__ import print_function
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from door_pointnet.dataset import LasDataset
from door_pointnet.model import PointNetCls, feature_transform_regularizer
from torch.nn import MSELoss, L1Loss
from tqdm import tqdm
from utils.log import log_loss

os.system('color')


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--train_dataset', type=str, required=True, help="Training dataset path")
parser.add_argument('--test_dataset', type=str, required=True, help="Testing dataset path")
parser.add_argument('--dataset_type', type=str, default='las', help="dataset type")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--log', type=str, help="Path to a log of the training and testing loss")
parser.add_argument('--augment', action='store_true', help='Perform data augmentation')

using_cuda = torch.cuda.is_available()
if using_cuda:
    print('We are using cuda')
else:
    print('We are NOT using cuda')

opt = parser.parse_args()
print(opt)

log_file = os.path.join('logs', opt.log) if opt.log else ''

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset, test_dataset = None, None
point_attributes = ['x', 'y']
if opt.dataset_type == 'las':
    dataset = LasDataset(
        root=opt.train_dataset,
        npoints=opt.num_points,
        point_attribs=point_attributes,
        data_augmentation=opt.augment)

    test_dataset = LasDataset(
        root=opt.test_dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False,
        point_attribs=point_attributes)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print('Number of training/testing items:', len(dataset), '/', len(test_dataset))
output_dim = len(dataset.output_names)
print('Output names', dataset.output_names)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=output_dim, feature_transform=opt.feature_transform, point_dim=dataset.point_dim)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
if using_cuda:
    classifier.cuda()

num_batch = len(dataset) / opt.batchSize
loss_function = MSELoss(reduction='sum')

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        points, target = data
        points = points.transpose(2, 1)
        if using_cuda:
            points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        if points.shape[0] == 1:
            continue
        pred, _, trans_feat = classifier(points)
        loss = loss_function(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))
        if opt.log:
            log_loss('train', epoch, loss.item(), opt.log)

        if i % 10 == 0:
            _, data = next(enumerate(testdataloader, 0))
            points, target = data
            points = points.transpose(2, 1)
            if using_cuda:
                points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = loss_function(pred, target)
            print('[%d: %d/%d] %s loss: %f' % (epoch, i, num_batch, blue('test'), loss.item()))
            if opt.log:
                log_loss('test', epoch, loss.item(), opt.log)
        scheduler.step()

    if epoch % 5:
        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

total_loss = 0
total_testset = 0
for i, data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    points = points.transpose(2, 1)
    if using_cuda:
        points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    loss = loss_function(pred, target)
    total_loss += loss
    total_testset += points.size()[0]
    if opt.log:
        log_loss('final_test', 0, loss.item(), opt.log)

print("final average loss {}".format(total_loss / float(total_testset)))
