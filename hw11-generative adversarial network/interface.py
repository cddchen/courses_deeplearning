import os.path
import torch
import torchvision.utils
import matplotlib.pyplot as plt
from model import Generator, Discriminator
from utils import set_device, same_seeds
from torch.autograd import Variable


if __name__ == '__main__':
    batch_size = 64
    z_dim = 100
    lr = 1e-4
    n_epoch = 30
    G = Generator(z_dim)
    print('Setting cuda&cpu...')
    device = torch.device('cpu')
    n_gpu = 0
    gpu_ids = None
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        gpu_ids = list(range(0, n_gpu))
        device = torch.device('cuda')
        if n_gpu > 1:
            G = torch.nn.DataParallel(
                G, device_ids=gpu_ids, output_device=gpu_ids[-1]
            )
            G = G.to(device)
            print('-> GPU training available! Training will use GPU(s) {}'.format(gpu_ids))
    G.load_state_dict(torch.load('checkpoints/nsgan_g_100.pth'))
    G.eval()
    # generate images and save the result
    n_output = 20
    z_sample = Variable(torch.randn(n_output, z_dim)).to(device)
    imgs_samples = (G(z_sample).data + 1) / 2.0
    save_dir = 'logs'
    filename = os.path.join(save_dir, 'result.jpg')
    torchvision.utils.save_image(imgs_samples, filename, nrow=10)
    # show image
    grid_img = torchvision.utils.make_grid(imgs_samples.cpu(), nrow=10)
    plt.figure(figsize=(10,10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
