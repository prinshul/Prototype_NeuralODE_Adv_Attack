from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method as FGSM
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent as PGD


"""Training Script"""
import os
import shutil
import numpy as np
import pdb
import random

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from tqdm import tqdm

if __name__ == '__main__':
    from util.utils import set_seed, CLASS_LABELS, date
    from config import ex
    from tensorboardX import SummaryWriter
    from dataloaders_medical.common import *
    from dataloaders_medical.prostate import *
    from models.encoder import SupportEncoder, QueryEncoder
    from models.decoder import Decoder
else:
    from .util.utils import set_seed, CLASS_LABELS, date
    from .config import ex
    from tensorboardX import SummaryWriter
    from .dataloaders_medical.common import *
    from .dataloaders_medical.prostate import *
    from .models.encoder import SupportEncoder, QueryEncoder
    from .models.decoder import Decoder

def overlay_color(img, mask, label, scale=50):
    """
    :param img: [1, 224, 224]
    :param mask: [1, 224, 224]
    :param label: [1, 224, 224]
    :return:
    """
    # pdb.set_trace()
    scale = np.mean(img.cpu().numpy())
    mask = mask[0]
    label = label[0]
    zeros = torch.zeros_like(mask)
    zeros = [zeros for _ in range(3)]
    zeros[0] = mask
    mask = torch.stack(zeros,dim=0)
    zeros[1] = label
    label = torch.stack(zeros,dim=0)
    img_3ch = torch.cat([img,img,img],dim=0)
    masked = img_3ch+mask.float()*scale+label.float()*scale
    return [masked]

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    device = torch.device(f"cuda:{_config['gpu_id']}")

    resize_dim = _config['input_size']
    encoded_h = int(resize_dim[0] / 2**_config['n_pool'])
    encoded_w = int(resize_dim[1] / 2**_config['n_pool'])

    s_encoder = SupportEncoder(_config['path']['init_path'], device)#.to(device)
    q_encoder = QueryEncoder(_config['path']['init_path'], device)#.to(device)
    decoder = Decoder(input_res=(encoded_h, encoded_w), output_res=resize_dim).to(device)

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'prostate':
        make_data = meta_data
    else:
        raise ValueError('Wrong config for dataset!')

    tr_dataset, val_dataset, ts_dataset = make_data(_config)
    trainloader = DataLoader(
        dataset=tr_dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['n_work'],
        pin_memory=False, #True load data while training gpu
        drop_last=True
    )

    # def perturb(x, y=None, tag="query"):
    # def perturb(q_encoder, decoder, s_xi_encode_avg, q_yi, x):
    def perturb(q_encoder, decoder, s_xi, q_xi, q_yi, to_attack):
        def wrapper_func(x):
            nonlocal q_xi, s_xi
            if to_attack == "s":
                s_xi_loc = x
                q_xi_loc = q_xi
            else:
                q_xi_loc = x
                s_xi_loc = s_xi
            s_xi = s_xi_loc
            q_xi = q_xi_loc
            # print(to_attack)
            s_x_merge = s_xi.view(s_xi.size(0) * s_xi.size(1), 1, 224, 224)
            s_y_merge = s_yi.view(s_yi.size(0) * s_yi.size(1), 1, 224, 224)
            s_xi_encode_merge, _ = s_encoder(s_x_merge, s_y_merge)  # [B*S, 512, w, h]

            s_xi_encode = s_xi_encode_merge.view(s_yi.size(0), s_yi.size(1), 512, encoded_w, encoded_h)
            s_xi_encode_avg = torch.mean(s_xi_encode, dim=1)

            q_xi_encode, q_ft_list = q_encoder(q_xi)
            sq_xi = torch.cat((s_xi_encode_avg, q_xi_encode), dim=1)
            yhati = decoder(sq_xi, q_ft_list)  # [B, 1, 224, 224]
            return yhati
        if to_attack == "s":
            x = s_xi
        else:
            x = q_xi
        if _config["attack"].upper() == "FGSM":
            init_shape = x.shape
            x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
            if _config["attack_eps"] == "random":
                epsilon = np.random.uniform(0.01,0.05)
            else:
                epsilon = _config["attack_eps"]
            x = FGSM(wrapper_func, x, epsilon, np.inf, y=q_yi)
            x = x.clone().detach()
            x = x.view(init_shape)
        elif _config["attack"].upper() == "PGD":
            init_shape = x.shape
            x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
            if _config["attack_eps"] == "random":
                epsilon = np.random.uniform(0.01,0.05)
            else:
                epsilon = _config["attack_eps"]
            x = PGD(wrapper_func, x, epsilon, 0.01, 40, np.inf, y=q_yi.to(torch.long))
            x = x.clone().detach()
            x = x.view(init_shape)
        else:
            pass
        return x

    _log.info('###### Set optimizer ######')
    print(_config['optim'])
    optimizer = torch.optim.Adam(#list(initializer.parameters()) +
                                 list(s_encoder.parameters()) +
                                 list(q_encoder.parameters()) +
                                 list(decoder.parameters()),
                                 _config['optim']['lr'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    pos_weight = torch.tensor([0.3 , 1], dtype=torch.float).to(device)
    criterion = nn.BCELoss()

    if _config['record']:  ## tensorboard visualization
        _log.info('###### define tensorboard writer #####')
        _log.info(f'##### board/train_{_config["board"]}_{date()}')
        writer = SummaryWriter(f'board/train_{_config["board"]}_{date()}')

    iter_n_train = len(trainloader)
    _log.info('###### Training ######')
    for i_epoch in tqdm(range(_config['n_steps'])):
        loss_epoch = 0
        blank = torch.zeros([1, 224, 224]).to(device)

        for i_iter, sample_train in enumerate(trainloader):
            adversarial_batch = np.random.uniform() > 0.5
            sq_choice = np.random.uniform()
            if sq_choice > 0.5:
                to_attack = "s"
            else:
                to_attack = "q"
            ## training stage
            optimizer.zero_grad()
            s_x = sample_train['s_x'].to(device)  # [B, Support, slice_num, 1, 224, 224]
            s_y = sample_train['s_y'].to(device)  # [B, Support, slice_num, 1, 224, 224]
            q_x = sample_train['q_x'].to(device) #[B, slice_num, 1, 224, 224]
            q_y = sample_train['q_y'].to(device) #[B, slice_num, 1, 224, 224]
            # loss_per_video = 0.0
            s_xi = s_x[:, :, 0, :, :, :]  # [B, Support, 1, 224, 224]
            s_yi = s_y[:, :, 0, :, :, :]
            
            q_xi = q_x[:, 0, :, :, :]
            q_yi = q_y[:, 0, :, :, :]


            if adversarial_batch:
                perturbed = perturb(q_encoder, decoder, s_xi.clone().detach(), q_xi.clone().detach(), q_yi, to_attack)
                if to_attack == "s":
                    s_xi = perturbed
                else:
                    q_xi = perturbed

            # for s_idx in range(_config["n_shot"]):
            s_x_merge = s_xi.view(s_xi.size(0) * s_xi.size(1), 1, 224, 224)
            s_y_merge = s_yi.view(s_yi.size(0) * s_yi.size(1), 1, 224, 224)
            s_xi_encode_merge, _ = s_encoder(s_x_merge, s_y_merge)  # [B*S, 512, w, h]

            s_xi_encode = s_xi_encode_merge.view(s_yi.size(0), s_yi.size(1), 512, encoded_w, encoded_h)
            s_xi_encode_avg = torch.mean(s_xi_encode, dim=1)
            # s_xi_encode, _ = s_encoder(s_xi, s_yi)  # [B, 512, w, h]


            q_xi_encode, q_ft_list = q_encoder(q_xi)
            sq_xi = torch.cat((s_xi_encode_avg, q_xi_encode), dim=1)
            yhati = decoder(sq_xi, q_ft_list)  # [B, 1, 224, 224]
            loss = criterion(yhati, q_yi)
            # loss_per_video += loss
            # loss_per_video.backward()
            loss.backward()
            optimizer.step()
            loss_epoch += loss
            print(f"train, iter:{i_iter}/{iter_n_train}, iter_loss:{loss}", end='\r')

            if _config['record'] and i_iter == 0:
                batch_i = 0
                frames = []
                frames += overlay_color(q_xi[batch_i], yhati[batch_i].round(), q_yi[batch_i], scale=_config['scale'])
                visual = make_grid(frames, normalize=True, nrow=2)
                writer.add_image("train/visual", visual, i_epoch)

                if _config['record'] and i_iter == 0:
                    batch_i = 0
                    frames = []
                    frames += overlay_color(q_xi[batch_i], yhati[batch_i].round(), q_yi[batch_i],
                                            scale=_config['scale'])
                    # frames += overlay_color(s_xi[batch_i], blank, s_yi[batch_i], scale=_config['scale'])
                    visual = make_grid(frames, normalize=True, nrow=5)
                    writer.add_image("valid/visual", visual, i_epoch)

        print(f"train - epoch:{i_epoch}/{_config['n_steps']}, epoch_loss:{loss_epoch}", end='\n')
        save_fname = f'{_run.observers[0].dir}/snapshots/last.pth'

        _run.log_scalar("training.loss", float(loss_epoch), i_epoch)
        if _config['record']:
            writer.add_scalar('loss/train_loss', loss_epoch, i_epoch)
        if i_epoch % 50 == 0 or i_epoch == _config['n_steps']-1:
            torch.save({
                    's_encoder': s_encoder.state_dict(),
                    'q_encoder': q_encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, save_fname
            )
        model_dir = "model_weights_adversarial"
        os.makedirs(model_dir, exist_ok=True)
        save_fname2 = os.path.join(model_dir, "./{}.pth".format(_config["model_name"]))
        torch.save({
                's_encoder': s_encoder.state_dict(),
                'q_encoder': q_encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_fname2
        )

    # writer.close()
