# from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method as FGSM
# from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent as PGD
from attacks import fast_gradient_method as FGSM
from attacks import projected_gradient_descent as PGD
from tqdm import tqdm
from smia import SMIA
# from asma import ASMA

"""Evaluation Script"""
import os
import shutil
import pdb
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
from torchvision.utils import make_grid
from math import isnan

from models.encoder import SupportEncoder, QueryEncoder
from models.decoder import Decoder

# from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS, get_bbox, date
from test_config import ex
from tensorboardX import SummaryWriter
from dataloaders_medical.prostate import *
import SimpleITK as sitk

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
    for source_file, _ in _run.experiment_info['sources']:
        os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                    exist_ok=True)
        _run.observers[0].save_file(source_file, f'source/{source_file}')
    shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{_config['gpu_id']}")

    _log.info('###### Create model ######')
    resize_dim = _config['input_size']
    encoded_h = int(resize_dim[0] / 2**_config['n_pool'])
    encoded_w = int(resize_dim[1] / 2**_config['n_pool'])

    s_encoder = SupportEncoder(_config['path']['init_path'], device)#.to(device)
    q_encoder = QueryEncoder(_config['path']['init_path'], device)#.to(device)
    decoder = Decoder(input_res=(encoded_h, encoded_w), output_res=resize_dim).to(device)

    checkpoint = torch.load(_config['snapshot'], map_location='cpu')
    s_encoder.load_state_dict(checkpoint['s_encoder'])
    q_encoder.load_state_dict(checkpoint['q_encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    print("no. of paramerters s encoder", sum(p.numel() for p in s_encoder.parameters()))
    print("no. of paramerters q encoder", sum(p.numel() for p in q_encoder.parameters()))
    print("no. of paramerters decoder", sum(p.numel() for p in decoder.parameters()))

    # initializer.eval()
    # encoder.eval()
    # convlstmcell.eval()
    # decoder.eval()

    _config["data_src"] = _config["data_srcs"][_config["dataset"]]

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    make_data = meta_data
    max_label = 1

    tr_dataset, val_dataset, ts_dataset = make_data(_config)
    testloader = DataLoader(
        dataset=ts_dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=_config['n_work'],
        pin_memory=False, # True
        drop_last=False
    )
    # all_samples = test_loader_Spleen()
    # all_samples = test_loader_Prostate()
    if _config['record']:
        _log.info('###### define tensorboard writer #####')
        board_name = f'board/test_{_config["board"]}_{date()}'
        writer = SummaryWriter(board_name)

    _log.info('###### Testing begins ######')
    # metric = Metric(max_label=max_label, n_runs=_config['n_runs'])
    img_cnt = 0
    # length = len(all_samples)
    length = len(testloader)
    img_lists = []
    pred_lists = []
    label_lists = []

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
            # x = x[0]
            if _config["attack_eps"] == "random":
                epsilon = np.random.uniform(0.01,0.05)
            else:
                epsilon = _config["attack_eps"]
            # print("goiong into FGSM, eps: {}".format(epsilon))
            x = FGSM(wrapper_func, x, epsilon, np.inf, y=q_yi, loss_fn=nn.BCELoss())
            x = x.view(init_shape)
            x = x.clone().detach()
        elif _config["attack"].upper() == "PGD":
            init_shape = x.shape
            x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
            if _config["attack_eps"] == "random":
                epsilon = np.random.uniform(0.01,0.05)
            else:
                epsilon = _config["attack_eps"]
            x = PGD(wrapper_func, x, epsilon, 0.01, 10, np.inf, y=q_yi, loss_fn=nn.BCELoss())
            x = x.view(init_shape)
            x = x.clone().detach()
        elif _config["attack"].upper() == "SMIA":
            if _config["attack_eps"] == "random":
                epsilon = np.random.uniform(0.01,0.05)
            else:
                epsilon = _config["attack_eps"]
            attack_model = SMIA(model = wrapper_func, epsilon=epsilon, loss_fn=nn.BCELoss())
            x = attack_model.perturb(x, y = q_yi, niters=10)
            x = x.detach()
        elif _config["attack"].upper() == "ASMA":
            init_shape = x.shape
            x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
            if _config["attack_eps"] == "random":
                epsilon = np.random.uniform(0.01,0.05)
            else:
                epsilon = _config["attack_eps"]
            x = PGD(wrapper_func, x, epsilon, 0.01, 40, np.inf, y=q_yi)
            x = x.clone().detach()
            x = x.view(init_shape)
        else:
            pass
        return x



    saves = {}
    for subj_idx in range(len(ts_dataset.get_cnts())):
        saves[subj_idx] = []


    loss_valid = 0
    batch_i = 0 # use only 1 batch size for testing

    for i, sample_test in enumerate(tqdm(testloader)): # even for upward, down for downward
        subj_idx, idx = ts_dataset.get_test_subj_idx(i)
        img_list = []
        pred_list = []
        label_list = []
        preds = []

        fnames = sample_test['q_fname']
        s_x = sample_test['s_x'].to(device)  # [B, slice_num, 1, 224, 224]
        s_y = sample_test['s_y'].to(device)  # [B, slice_num, 1, 224, 224]
        q_x = sample_test['q_x'].to(device)  # [B, slice_num, 1, 224, 224]
        q_y = sample_test['q_y'].to(device)  # [B, slice_num, 1, 224, 224]
        s_xi = s_x[:, :, 0, :, :, :]  # [B, Support, 1, 224, 224]
        s_yi = s_y[:, :, 0, :, :, :]
        s_yi = (s_yi>0).to(torch.float)

        q_xi = q_x[:, 0, :, :, :]
        q_yi = q_y[:, 0, :, :, :]
        q_yi = (q_yi>0).to(torch.float)

        if _config["to_attack"] == "s" or _config["to_attack"] == "q":
            perturbed = perturb(q_encoder, decoder, s_xi.clone().detach(), q_xi.clone().detach(), q_yi, to_attack = _config["to_attack"])
            if _config["to_attack"] == "s":
                s_xi = perturbed
            else:
                q_xi = perturbed

        with torch.no_grad():
            for s_idx in range(_config["n_shot"]):
                s_x_merge = s_xi.view(s_xi.size(0) * s_xi.size(1), 1, 224, 224)
                s_y_merge = s_yi.view(s_yi.size(0) * s_yi.size(1), 1, 224, 224)
                s_xi_encode_merge, _ = s_encoder(s_x_merge, s_y_merge)  # [B*S, 512, w, h]

            s_xi_encode = s_xi_encode_merge.view(s_yi.size(0), s_yi.size(1), 512, encoded_w, encoded_h)
            s_xi_encode_avg = torch.mean(s_xi_encode, dim=1)
            # s_xi_encode, _ = s_encoder(s_xi, s_yi)  # [B, 512, w, h]
            
            q_xi_encode, q_ft_list = q_encoder(q_xi)
            sq_xi = torch.cat((s_xi_encode_avg, q_xi_encode), dim=1)
            yhati = decoder(sq_xi, q_ft_list)  # [B, 1, 224, 224]

            preds.append(yhati.round())
            img_list.append(q_xi[batch_i].cpu().numpy())
            pred_list.append(yhati[batch_i].round().cpu().numpy())
            label_list.append(q_yi[batch_i].cpu().numpy())

            saves[subj_idx].append([subj_idx, idx, img_list, pred_list, label_list, fnames])
            # print(f"test, iter:{i}/{length} - {subj_idx}/{idx} \t\t", end='\r')
            img_lists.append(img_list)
            pred_lists.append(pred_list)
            label_lists.append(label_list)

    print("start computing dice similarities ... total ", len(saves))
    dice_similarities = []
    for subj_idx in range(len(saves)):
        imgs, preds, labels = [], [], []
        save_subj = saves[subj_idx]
        for i in range(len(save_subj)):
            # print(len(save_subj), len(save_subj)-q_slice_n+1, q_slice_n, i)
            subj_idx, idx, img_list, pred_list, label_list, fnames = save_subj[i]
            # print(subj_idx, idx, is_reverse, len(img_list))
            # print(i, is_reverse, is_reverse_next, is_flip)

            for j in range(len(img_list)):
                imgs.append(img_list[j])
                preds.append(pred_list[j])
                labels.append(label_list[j])

        # pdb.set_trace()
        img_arr = np.concatenate(imgs, axis=0)
        pred_arr = np.concatenate(preds, axis=0)
        label_arr = np.concatenate(labels, axis=0)
        # print(ts_dataset.slice_cnts[subj_idx] , len(imgs))
        # pdb.set_trace()
        dice = np.sum([label_arr * pred_arr]) * 2.0 / (np.sum(pred_arr) + np.sum(label_arr))
        dice_similarities.append(dice)
        # print(f"computing dice scores {subj_idx}/{10}", end='\n')

        if _config["save_vis"]:
            vis_save_root = "visualisations_final"
            vis_save_name = _config["log_name"].replace(" ", "_")
            vis_save_name.replace("_|_", "_")
            vis_save_name = vis_save_name + "_" + _config["attack"]
            if _config["attack"].upper() =="PGD" or _config["attack"].upper() =="FGSM":
                vis_save_name += "_" + str(_config["attack_eps"])
                vis_save_name += "_" + _config["to_attack"]
            for frame_id in range(0, len(save_subj)):
                vis_save_dir = os.path.join(vis_save_root, vis_save_name, str(subj_idx),  str(frame_id))
                os.makedirs(vis_save_dir, exist_ok=True)
                img = imgs[frame_id][0, ...]
                pred = preds[frame_id][0, ...]
                label = labels[frame_id][0, ...]
                cv2.imwrite(os.path.join(vis_save_dir, "query.png"), 255*img)
                cv2.imwrite(os.path.join(vis_save_dir, "label.png"), 255*label)
                cv2.imwrite(os.path.join(vis_save_dir, "prediction.png"), 255*pred)
                

        if _config['record']:
            frames = []
            for frame_id in range(0, len(save_subj)):
                frames += overlay_color(torch.tensor(imgs[frame_id]), torch.tensor(preds[frame_id]), torch.tensor(labels[frame_id]), scale=_config['scale'])
            visual = make_grid(frames, normalize=True, nrow=5)
            writer.add_image(f"test/{subj_idx}", visual, i)
            writer.add_scalar(f'dice_score/{i}', dice)

        if _config['save_sample']:
            ## only for internal test (BCV - MICCAI2015)
            sup_idx = _config['s_idx']
            target = _config['target']
            save_name = _config['save_name']
            dirs = ["gt", "pred", "input"]
            save_dir = f"./PANet/MICCAI2015/sample/fss1000_organ{target}_sup{sup_idx}_{save_name}"

            for dir in dirs:
                try:
                    os.makedirs(os.path.join(save_dir,dir))
                except:
                    pass

            subj_name = fnames[0][0].split("/")[-2]
            if target == 14:
                src_dir = "./Cervix/RawData/Training/img"
                orig_fname = f"{src_dir}/{subj_name}-Image.nii.gz"
                pass
            else:
                src_dir = "./Abdomen/RawData/Training/img"
                orig_fname = f"{src_dir}/img{subj_name}.nii.gz"

            itk = sitk.ReadImage(orig_fname)
            orig_spacing = itk.GetSpacing()

            label_arr = label_arr*2.0
            # label_arr = np.concatenate([np.zeros([1,256,256]), label_arr,np.zeros([1,256,256])])
            # pred_arr = np.concatenate([np.zeros([1,256,256]), pred_arr,np.zeros([1,256,256])])
            # img_arr = np.concatenate([np.zeros([1,256,256]), img_arr,np.zeros([1,256,256])])
            # pdb.set_trace()
            itk = sitk.GetImageFromArray(label_arr)
            itk.SetSpacing(orig_spacing)
            sitk.WriteImage(itk,f"{save_dir}/gt/{subj_idx}.nii.gz")
            itk = sitk.GetImageFromArray(pred_arr.astype(float))
            itk.SetSpacing(orig_spacing)
            sitk.WriteImage(itk,f"{save_dir}/pred/{subj_idx}.nii.gz")
            itk = sitk.GetImageFromArray(img_arr.astype(float))
            itk.SetSpacing(orig_spacing)
            sitk.WriteImage(itk,f"{save_dir}/input/{subj_idx}.nii.gz")


    print( _config["log_name"] + f" | test result \n n : {len(dice_similarities)}, mean dice score : \
    {np.mean(dice_similarities)} \n dice similarities : {dice_similarities}")

    with open("final_results.log", 'a') as f:
        f.write("\n" + _config["log_name"])
        f.write(" | Mean dice score : {:.4f}".format(np.mean(dice_similarities)))
        f.write("\n" + "="*60)

    if _config['record']:
        writer.add_scalar(f'dice_score/mean', np.mean(dice_similarities))

