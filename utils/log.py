import os.path as osp


def log_loss(category, value, log_filename):
    with open(log_filename, 'w') as f:
        if not osp.exists(log_filename):
            f.write('category,value')
        f.write(f'{category},{value}')
