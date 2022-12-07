import os.path as osp


def log_loss(category, epoch, value, log_filename):
    with open(log_filename, 'a') as f:
        if not osp.exists(log_filename):
            f.write('category,epoch,value')
        f.write(f'{category},{epoch},{value}')
