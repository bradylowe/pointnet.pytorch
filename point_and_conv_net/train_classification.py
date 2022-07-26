from __future__ import print_function
import argparse
import os
import random
import torch.optim as optim
import torch.utils.data
from point_and_conv_net.dataset import PointsAndSlices
from point_and_conv_net.model import PointnetPlusConv
from torch.nn import MSELoss
from tqdm import tqdm
from convnet.plot_data import plot_arrays
from convnet.dataset import RACK_SCALE

os.system('color')


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--datasets', type=str, required=True, nargs='+', help="Paths to datasets")

using_cuda = torch.cuda.is_available()
if using_cuda:
    print('We are using cuda')
else:
    print('We are NOT using cuda')

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PointsAndSlices(paths=opt.datasets, split='train')
test_dataset = PointsAndSlices(paths=opt.datasets, split='test')

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

classifier = PointnetPlusConv(image_resolution=dataset.resolution, n_slices=dataset.n_slices)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
if using_cuda:
    classifier.cuda()

num_batch = len(dataset) / opt.batchSize
loss_function = MSELoss(reduction='mean')

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        data, target = data
        if using_cuda:
            data[0], data[1], target = data[0].cuda(), data[1].cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred = classifier(data)
        loss = loss_function(pred, target)
        loss.backward()
        optimizer.step()
        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss * RACK_SCALE))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            data, target = data
            if using_cuda:
                data[0], data[1], target = data[0].cuda(), data[1].cuda(), target.cuda()
            classifier = classifier.eval()
            pred = classifier(data)
            loss = loss_function(pred, target)
            print('[%d: %d/%d] %s loss: %f' % (epoch, i, num_batch, blue('test'), loss * RACK_SCALE))
        scheduler.step()

    if epoch % 50 == 0:
        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))
        weights = [classifier.conv1a.weight.cpu().detach(),
                   classifier.conv1b.weight.cpu().detach()]
        old_weights = [w for w in weights]

        png_dir = os.path.abspath(os.path.join('.', 'weights_png'))
        if not os.path.isdir(png_dir):
            os.makedirs(png_dir)
        png_file = os.path.join(png_dir, f'rough_weights_epoch{epoch}_conv1.png')
        fig, _ = plot_arrays(weights[0][:64, 0, :, :], shape=(8, 8), show_labels=False)
        fig.savefig(png_file, pad_inches=0.1, dpi=1000)
        png_file = os.path.join(png_dir, f'fine_weights_epoch{epoch}_conv2.png')
        fig, _ = plot_arrays(weights[1][:64, 0, :, :], shape=(8, 8), show_labels=False)
        fig.savefig(png_file, pad_inches=0.1, dpi=1000)


total_loss = 0
for i, data in tqdm(enumerate(testdataloader, 0)):
    data, target = data
    if using_cuda:
        data[0], data[1], target = data[0].cuda(), data[1].cuda(), target.cuda()
    classifier = classifier.eval()
    pred = classifier(data)
    total_loss += loss_function(pred, target)

print("final average loss {}".format(total_loss * RACK_SCALE))
