"""
Fewshot Semantic Segmentation
"""
import pdb
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg import Encoder


class FewShotSeg(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, in_channels=1, pretrained_path=None):
        super().__init__()
        self.pretrained_path = pretrained_path

        # Encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', Encoder(in_channels, self.pretrained_path)),]))


    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, factor=1, return_feats=False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 1 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 1 x H x W], list of tensors
        """
        self.factor = factor
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)
        img_fts = self.encoder(imgs_concat)
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H x W
        # back_mask = torch.stack([torch.stack(way, dim=0)
        #                          for way in back_mask], dim=0)  # Wa x Sh x B x H x W

        back_mask = torch.ones_like(fore_mask) - fore_mask
        ###### Compute loss ######
        outputs = []
        all_prototypes = []
        all_fg_prototypess = []
        for epi in range(batch_size):
            ###### Extract prototype ######
            supp_fg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             back_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]

            ###### Obtain the prototypes######
            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)

            ###### Compute the distance ######
            prototypes = [bg_prototype,] + fg_prototypes
            dist = [self.calDist(qry_fts[:, epi], prototype) for prototype in prototypes]
            all_prototypes += prototypes
            all_fg_prototypess += fg_prototypes
            
            pred = torch.stack(dist, dim=1)  # N x (1 + Wa) x H' x W'
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))

        all_prototypes = torch.stack(all_prototypes,  dim=0)
        all_fg_prototypess = torch.stack(all_fg_prototypess,  dim=0)
        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        if return_feats:
            return output, all_prototypes, qry_fts
        return output, all_fg_prototypess

    def get_sup_fore(self, sup, fore_mask):
        # print(len(sup), len(sup[0]), len(sup[0][0]), len(sup[0][0][0]), len(sup[0][0][0][0]))
        batch_size = sup[0].shape[0]
        # print("batch size: ", batch_size)
        # sup = torch.cat([torch.cat(sample, dim=0) for sample in sup])
        sup = torch.cat(sup, dim=0)
        sup = sup.squeeze(1)
        # print(sup.shape)
        supp_fts = self.encoder(sup)
        # print(len(fore_mask), len(fore_mask[0]), len(fore_mask[0][0]), fore_mask[0][0][0].shape)
        num_samples = (supp_fts.shape[0]//batch_size)
        # print(num_samples)
        # print(supp_fts.shape, fore_mask.shape)
        supp_fg_fts = [self.getFeatures(supp_fts[i, ...].unsqueeze(0),  fore_mask[i%num_samples][0][i//num_samples].unsqueeze(0)) for i in range(supp_fts.shape[0])]
        supp_fg_fts = [torch.cat(supp_fg_fts[i*batch_size: (i+1)*batch_size], dim=0) for i in range(len(supp_fg_fts)//batch_size)]
        return supp_fg_fts


    def calDist(self, fts, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
        return dist


    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        # print(fts.shape, mask.shape)
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
            / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # 1 x C
        return masked_fts


    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        if n_shots > self.factor:
            n_shots = n_shots // self.factor
        fg_prototypes = [sum(way[:n_shots]) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way[:n_shots]) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype
