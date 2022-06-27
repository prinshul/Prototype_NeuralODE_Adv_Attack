from attacks import fast_gradient_method as FGSM
from attacks import projected_gradient_descent as PGD
from smia import SMIA
from asma import ASMA
import torch.nn.functional as F

"""Evaluation Script"""
import os
import shutil
import pdb
import tqdm
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid

from models.fewshot import FewShotSeg
from models.ode import FewShotSegOde
from util.utils import set_seed, CLASS_LABELS, get_bbox, date
from test_config import ex

from tensorboardX import SummaryWriter
from dataloaders_medical.prostate import *
import SimpleITK as sitk
from tqdm import tqdm
import cv2


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


    _log.info('###### Create model ######')
    if _config["use_ode"]:
        model_orig = FewShotSegOde(pretrained_path=_config['path']['init_path'], pretrained_ode=_config["pretrain_ode"], ode_layers=_config["ode_layers"], ode_time=_config["ode_time"])
    else:
        model_orig = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model_orig.cuda(), device_ids=[_config['gpu_id'],])
    if not _config['notrain']:
        model.load_state_dict(torch.load(_config['snapshot'], map_location='cpu'))
    model.eval()


    # _config["data_src"] = _config["data_srcs"][_config["dataset"]][str(_config["server"])]

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
    print("length", length)

    def perturb(s_x, s_y_fg, s_y_bg, q_x, y, to_attack="q"):
        def wrapper_fn(model):
            def fun(x):
                nonlocal q_x, s_x
                if to_attack == "s":
                    s_x_loc = x
                    q_x_loc = q_x
                else:
                    q_x_loc = x
                    s_x_loc = s_x
                s_x = s_x_loc
                q_x = q_x_loc

                s_xs = [[s_x[:,shot, ...] for shot in range(_config["n_shot"])]]
                s_y_fgs = [[s_y_fg[:,shot, ...] for shot in range(_config["n_shot"])]]
                s_y_bgs = [[s_y_bg[:,shot, ...] for shot in range(_config["n_shot"])]]
                q_xs = [q_x]
                return model(s_xs, s_y_fgs, s_y_bgs, q_xs)
            return fun
        if to_attack == "s":
            x = s_x
        else:
            x = q_x
        if _config["attack"].upper() == "FGSM":
            local_model = wrapper_fn(model_orig)
            x = FGSM(local_model, x, _config["attack_eps"], np.inf, y=y.view(-1, y.shape[-2], y.shape[-1]).to(torch.long))
            x = x.detach()
        elif _config["attack"].upper() == "PGD":
            local_model = wrapper_fn(model_orig)
            x = PGD(local_model, x, _config["attack_eps"], 0.01, 40, np.inf, y=y.view(-1, y.shape[-2], y.shape[-1]).to(torch.long))
            x = x.detach()
        elif _config["attack"].upper() == "SMIA":
            attack_model = SMIA(model = wrapper_fn(model_orig), epsilon=_config["attack_eps"], loss_fn=torch.nn.CrossEntropyLoss())
            x = attack_model.perturb(x, niters=40, y = y.view(-1, y.shape[-2], y.shape[-1]).to(torch.long))
            x = x.detach()
        else:
            pass
        return x


    saves = {}
    for subj_idx in range(len(ts_dataset.get_cnts())):
        saves[subj_idx] = []

    
    loss_valid = 0
    batch_i = 0 # use only 1 batch size for testing
    printed = False
    all_prototypes  = []
    all_query_feats_orig = []
    all_query_feats_pert = []
    for i, sample_test in enumerate(tqdm(testloader)): # even for upward, down for downward
        
        subj_idx, idx = ts_dataset.get_test_subj_idx(i)
        img_list = []
        pred_list = []
        label_list = []
        preds = []
        fnames = sample_test['q_fname']
        

        s_x_orig = sample_test['s_x'].cuda()  # [B, Support, slice_num=1, 1, 224, 224]
        s_x = s_x_orig.squeeze(2)  # [B, Support, 1, 224, 224]
        s_y_fg_orig = sample_test['s_y'].cuda()  # [B, Support, slice_num, 1, 224, 224]
        s_y_fg = s_y_fg_orig.squeeze(2)  # [B, Support, 1, 224, 224]
        s_y_fg = s_y_fg.squeeze(2)  # [B, Support, 224, 224]
        s_y_bg = torch.ones_like(s_y_fg) - s_y_fg
        q_x_orig = sample_test['q_x'].cuda()  # [B, slice_num, 1, 224, 224]
        

        q_x = q_x_orig.squeeze(1)  # [B, 1, 224, 224]
        q_y_orig = sample_test['q_y'].cuda()  # [B, slice_num, 1, 224, 224]
        q_y = q_y_orig.squeeze(1)  # [B, 1, 224, 224]
        q_y = q_y.squeeze(1).long()  # [B, 224, 224]

        s_xs_orig = [[s_x[:, shot, ...] for shot in range(_config["n_shot"])]]
        s_y_fgs_orig = [[s_y_fg[:, shot, ...] for shot in range(_config["n_shot"])]]
        s_y_bgs_orig = [[s_y_bg[:, shot, ...] for shot in range(_config["n_shot"])]]
        q_xs_orig = [q_x]
        
        if _config["to_attack"] == "s" or _config["to_attack"] == "q":
            perturbed = perturb(s_x, s_y_fg, s_y_bg, q_x, q_y_orig, _config["to_attack"])
            if _config["to_attack"] == "s":
                s_x = perturbed
            else:
                q_x = perturbed
        
        s_xs = [[s_x[:, shot, ...] for shot in range(_config["n_shot"])]]
        s_y_fgs = [[s_y_fg[:, shot, ...] for shot in range(_config["n_shot"])]]
        s_y_bgs = [[s_y_bg[:, shot, ...] for shot in range(_config["n_shot"])]]
        
        q_xs = [q_x]

        with torch.no_grad():
            q_yhat_orig, batch_prototypes_pert, batch_query_feats_pert = model(s_xs, s_y_fgs, s_y_bgs, q_xs, return_feats=True)
            q_yhat_pert, batch_prototypes_orig, batch_query_feats_orig = model(s_xs_orig, s_y_fgs_orig, s_y_bgs_orig, q_xs_orig, return_feats=True)
        all_prototypes.append(batch_prototypes_orig)
        all_query_feats_orig.append(batch_query_feats_pert)
        all_query_feats_pert.append(batch_query_feats_orig)

        q_yhat = q_yhat_pert.argmax(dim=1)
        if not printed:
            print(q_yhat.shape)
            printed  = True
        q_yhat = q_yhat.unsqueeze(1)

        preds.append(q_yhat)
        # img_list.append(q_x_orig[batch_i,0].cpu().numpy())
        img_list.append(q_x.view(1, 224, 224).cpu().numpy())
        pred_list.append(q_yhat[batch_i].cpu().numpy())
        label_list.append(q_y_orig[batch_i,0].cpu().numpy())

        saves[subj_idx].append([subj_idx, idx, img_list, pred_list, label_list, fnames, batch_query_feats_orig, batch_query_feats_pert, q_yhat_orig, q_yhat_pert, batch_prototypes_orig])
        # print(f"test, iter:{i}/{length} - {subj_idx}/{idx} \t\t", end='\r')
        img_lists.append(img_list)
        pred_lists.append(pred_list)
        label_lists.append(label_list)


    print("start computing dice similarities ... total ", len(saves))
    dice_similarities = []
    best_dice = 0
    best_query = None
    for subj_idx in range(len(saves)):
        imgs, preds, labels = [], [], []
        subj_query_feats_orig, subj_query_feats_pert = [],  []
        subj_q_hat_orig, subj_q_hat_pert = [], []
        subj_prototypes_orig = []
        save_subj = saves[subj_idx]
        for i in range(len(save_subj)):
            # print(len(save_subj), len(save_subj)-q_slice_n+1, q_slice_n, i)
            subj_idx, idx, img_list, pred_list, label_list, fnames, batch_query_feats_orig, batch_query_feats_pert, q_yhat_orig, q_yhat_pert, batch_prototypes_orig = save_subj[i]
            # print(subj_idx, idx, is_reverse, len(img_list))
            # print(i, is_reverse, is_reverse_next, is_flip)

            for j in range(len(img_list)):
                imgs.append(img_list[j])
                preds.append(pred_list[j])
                labels.append(label_list[j])
                subj_query_feats_orig.append(batch_query_feats_orig.cpu().numpy())
                subj_query_feats_pert.append(batch_query_feats_pert.cpu().numpy())
                subj_q_hat_orig.append(q_yhat_orig.cpu().numpy())
                subj_q_hat_pert.append(q_yhat_pert.cpu().numpy())
                subj_prototypes_orig.append(batch_prototypes_orig)

        # pdb.set_trace()
        img_arr = np.concatenate(imgs, axis=0)
        pred_arr = np.concatenate(preds, axis=0)
        label_arr = np.concatenate(labels, axis=0)
        

        # pdb.set_trace()
        # print(ts_dataset.slice_cnts[subj_idx] , len(imgs))
        # pdb.set_trace()
        dice = np.sum([label_arr * pred_arr]) * 2.0 / (np.sum(pred_arr) + np.sum(label_arr))
        dice_similarities.append(dice)

        if dice >= best_dice:
            subj_query_feats_orig_arr = np.concatenate(subj_query_feats_orig, axis=0)
            subj_query_feats_pert_arr = np.concatenate(subj_query_feats_pert, axis=0)
            subj_q_hat_orig_arr = np.concatenate(subj_q_hat_orig, axis=0)
            subj_q_hat_pert_arr = np.concatenate(subj_q_hat_pert, axis=0)
            best_query = (subj_query_feats_orig_arr, subj_query_feats_pert_arr, subj_q_hat_orig_arr, subj_q_hat_pert_arr, subj_prototypes_orig)
        # print(f"computing dice scores {subj_idx}/{10}", end='\n')
        _config["save_vis"] = True
        if _config["save_vis"]:
            vis_save_root = "visualisations"
            vis_save_name = _config["snapshot"].split("/")[-1].split(".")[0]
            vis_save_name = vis_save_name + "_" + _config["attack"]
            if _config["attack"].upper() =="PGD" or _config["attack"].upper() =="FGSM" or _config["attack"].upper() =="SMIA" or _config["attack"].upper() =="ASMA":
                vis_save_name += "_" + str(_config["attack_eps"])
                vis_save_name += "_" + _config["to_attack"]
            for frame_id in range(0, len(save_subj)):
                vis_save_dir = os.path.join(vis_save_root, vis_save_name, str(frame_id))
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
                frames += overlay_color(torch.tensor(imgs[frame_id]), torch.tensor(preds[frame_id]).float(), torch.tensor(labels[frame_id]))
            visual = make_grid(frames, normalize=True, nrow=5)
            writer.add_image(f"test/{subj_idx}", visual, i)
            writer.add_scalar(f'dice_score/{i}', dice)

        if _config['save_sample']:
            ## only for internal test (BCV - MICCAI2015)
            sup_idx = _config['s_idx']
            target = _config['target']
            save_name = _config['save_name']
            dirs = ["gt", "pred", "input"]
            save_dir = f"../sample/panet_organ{target}_sup{sup_idx}_{save_name}"

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
            itk = sitk.GetImageFromArray(label_arr)
            itk.SetSpacing(orig_spacing)
            sitk.WriteImage(itk,f"{save_dir}/gt/{subj_idx}.nii.gz")
            itk = sitk.GetImageFromArray(pred_arr.astype(float))
            itk.SetSpacing(orig_spacing)
            sitk.WriteImage(itk,f"{save_dir}/pred/{subj_idx}.nii.gz")
            itk = sitk.GetImageFromArray(img_arr)
            itk.SetSpacing(orig_spacing)
            sitk.WriteImage(itk,f"{save_dir}/input/{subj_idx}.nii.gz")



    all_prototypes = torch.cat(all_prototypes, dim=0).view(-1, 512).cpu().numpy()
    print(all_prototypes.shape)
    # np.save("panet_prototypes_fgsm.npy", all_prototypes)


    query_feats_orig, query_feats_pert, q_hat_orig, q_hat_pert, prototypes = best_query
    # print(prototypes)
    print(len(prototypes), prototypes[0].shape)
    all_prototypes = torch.cat(prototypes, dim=0).view(-1, 512).cpu().numpy()
    all_prototypes  = all_prototypes[:all_prototypes.shape[0]//2, ...]
    print(all_prototypes.shape)
    print(q_hat_orig.shape, q_hat_pert.shape)
    print(subj_query_feats_orig_arr.shape, subj_query_feats_pert_arr.shape)
    q_hat_orig = F.interpolate(torch.Tensor(q_hat_orig), size=(32, 32), mode='bilinear').cpu().numpy()
    q_hat_pert = F.interpolate(torch.Tensor(q_hat_pert), size=(32, 32), mode='bilinear').cpu().numpy()
    # sorted_indices_fg = np.argsort(q_hat_orig[:, 1, :, :].reshape(-1))
    # sorted_indices_bg = np.argsort(q_hat_orig[:, 0, :, :].reshape(-1))
    opt = q_hat_pert + q_hat_orig
    sorted_indices_fg = np.argsort(opt[:, 1, :, :].reshape(-1))
    sorted_indices_bg = np.argsort(opt[:, 0, :, :].reshape(-1))

    top_k = 5
    best_fg_queries_orig = query_feats_orig.reshape(-1,  512)[sorted_indices_fg[-top_k:], :]
    best_bg_queries_orig = query_feats_orig.reshape(-1,  512)[sorted_indices_bg[-top_k:], :]
    best_fg_queries_pert = query_feats_pert.reshape(-1,  512)[sorted_indices_fg[-top_k:], :]
    best_bg_queries_pert = query_feats_pert.reshape(-1,  512)[sorted_indices_bg[-top_k:], :]
    # print(best_fg_queries.shape,  best_bg_queries.shape)


    all_feats = np.concatenate([all_prototypes, best_fg_queries_orig, best_bg_queries_orig, best_fg_queries_pert, best_bg_queries_pert], axis=0)


    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    all_feats_reduced = tsne.fit_transform(all_feats)

    print(all_feats_reduced.shape)
    plt.scatter(all_feats_reduced[-2*top_k:, 0], all_feats_reduced[-2*top_k:, 1], c="r", label="perturbed queries")
    for i in range(2*top_k):
        plt.arrow(all_feats_reduced[-4*top_k  + i, 0], all_feats_reduced[-4*top_k  + i, 1], -all_feats_reduced[-4*top_k  + i, 0] + all_feats_reduced[-2*top_k  + i, 0], -all_feats_reduced[-4*top_k  + i, 1] + all_feats_reduced[-2*top_k  + i, 1], width = 0.05, head_width = 0.2)

    plt.scatter(all_feats_reduced[:-4*top_k:2, 0], all_feats_reduced[:-4*top_k:2, 1], c="g", label="fg prototypes")
    plt.scatter(all_feats_reduced[1:-4*top_k:2, 0], all_feats_reduced[1:-4*top_k:2, 1], c="y", label="bg prototypes")
    # plt.scatter(all_feats_reduced[:-80, 0], all_feats_reduced[:-80, 1], c="g", label="prototypes")
    plt.scatter(all_feats_reduced[-4*top_k:-2*top_k, 0], all_feats_reduced[-4*top_k:-2*top_k, 1], c="b", label="clean queries")
    
    plt.legend()

    plt.savefig("tsne_subj_pert_best_deltas_{}.png".format(top_k), dpi=300)



    print(f"test result \n n : {len(dice_similarities)}, mean dice score : \
    {np.mean(dice_similarities)} \n dice similarities : {dice_similarities}")
    

    with open("test_results.log", 'a') as f:
        f.write("\n" + _config["log_name"])
        f.write(" | Mean dice score : {:.4f}".format(np.mean(dice_similarities)))
        f.write("\n" + "="*60)

    if _config['record']:
        writer.add_scalar(f'dice_score/mean', np.mean(dice_similarities))
