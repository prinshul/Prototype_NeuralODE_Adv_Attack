"""Experiment Configuration"""
import os
import re
import glob
import itertools

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('FSS1000', save_git_info=False)
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './util', './dataloaders_medical']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)

    # "Organs" : ["background",
    #           "spleen",
    #           "right kidney",
    #           "left kidney",
    #           "gallbladder",
    #           "esophagus",
    #           "liver",
    #           "stomach",
    #           "aorta",
    #           "inferior vana cava",
    #           "portal vein & splenic vein",
    #           "pancreas",
    #           "right adrenal gland",
    #           "left adrenal gland",
    #           ],

@ex.config
def cfg():
    """Default configurations"""

    server="144" #202
    # size = 320
    size = 256
    input_size = (size, size) # 419? 480!
    seed = 1234
    cuda_visable = '0, 1, 2, 3, 4, 5, 6, 7'
    gpu_id = 0
    mode = 'train' # 'train' or 'test'
    record = False
    scale = 1.0

    n_shot = 1
    s_idx = 0

    n_pool = 3 # 4 - number of pooling
    target = 6
    add_target = 0

    external_test = "None"  # "decathlon" # "CT_ORG"
    if external_test == "None":
        internal_test = True
    else:
        internal_test = False

    model_name = "test"
    ### Attack configs
    # attack = "PGD"
    attack = "FGSM"
    # attack = "None"
    # attack_eps = 0.03
    attack_eps = "random"
    attack_support = False
    attack_query = True
    attack_visualise = False

    if mode == 'train':
        lr_milestones = [50*i for i in range(1,3)]
        n_iter = 500
        dataset = 'prostate'  # 'VOC' or 'COCO'
        n_steps = 300
        n_work = 1
        label_sets = 0
        batch_size = 5
        print_interval = 500
        validation_interval = 500
        save_pred_every = 10000
        val_cnt = 100

        task = {
            'n_ways': 1,
            'n_shots': n_shot,
            'n_queries': 1,
        }

        model = {
            'align': False,
            # 'align': True,
        }
        optim = {
            'lr': 1e-4,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        }

    elif mode == 'test':
        save_sample = False
        save_name = ""

        is_test=True
        dataset = 'prostate'  # 'VOC' or 'COCO'
        notrain = False
        # snapshot = './runs/PANet_VOC_sets_0_1way_1shot_[train]/1/snapshots/30000.pth'
        snapshot = 'PANet_prostate_sets_0_1way_1shot_train.pth'
        n_iter = 1
        n_runs = 1
        n_steps = 1000
        batch_size = 1
        scribble_dilation = 0
        bbox = False
        scribble = False
        HE = False
        # Set dataset config from the snapshot string

        # Set model config from the snapshot string
        model = {}
        for key in ['align',]:
            model[key] = key in snapshot
        label_sets = int(snapshot.split('_sets_')[1][0])

        # Set task config from the snapshot string
        task = {
            'n_ways': 1,
            'n_shots': n_shot,
            'n_queries': 1,
        }

    else:
        raise ValueError('Wrong configuration for "mode" !')


    #exp_str = '_'.join([
       # mode,
    #])
    exp_str = '_'.join(
        [dataset,]
        + [key for key, value in model.items() if value]
        + [f'sets_{label_sets}', f'{task["n_ways"]}way_{task["n_shots"]}shot_{mode}'])
    data_srcs = {
        "BCV": "./organ_data/BCV/Training_2d_nocrop",
        "CTORG": "./organ_data/CT_ORG/Training_2d_nocrop",
        "Decathlon": "./organ_data/Decathlon/Training_2d_nocrop",
    }

    data_src = data_srcs[dataset]

    path = {
        'log_dir': './runs',
        # 'init_path': None,
        'init_path': '/./pretrained_model/vgg16-397923af.pth',
    }

    ### configuration for Medical Image Test
    modal_index = 0 #["flair","t1","t1ce","t2"]
    mask_index = 1 #[1, 2, 4]
    board=""

@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook function to add observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
