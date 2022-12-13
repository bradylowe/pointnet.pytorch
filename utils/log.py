import os


class LogLoss:

    def __init__(self, log_filename):
        self.filename = log_filename
        if log_filename:
            if os.path.exists(log_filename):
                os.remove(log_filename)
            self.f = open(log_filename, 'a')
            self.f.write('category,epoch,value\n')
        else:
            self.f = None

    def write(self, category, epoch, value):
        self.f.write(f'{category},{epoch},{value}\n')

    def __bool__(self):
        return bool(self.filename) and not self.f.closed

    def __exit__(self):
        self.f.close()
