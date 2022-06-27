import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import math
from torch.autograd import Variable
from torch import cuda
import scipy.stats as st

def gkern(kernlen = 3, nsig = 1):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(-1,x)
    kern2d = st.norm.pdf(-1,x)
    kernel_raw = np.outer(kern1d, kern2d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def blur(tensor_image, epsilon, stack_kerne):        
    min_batch=tensor_image.shape[0]        
    channels=tensor_image.shape[1]        
    out_channel=channels       
    kernel=torch.FloatTensor(stack_kerne).cuda()	
    weight = nn.Parameter(data=kernel, requires_grad=False)         
    init_shape = tensor_image.shape
    data_grad=F.conv2d(tensor_image.view(-1, init_shape[-3], init_shape[-2], init_shape[-1]),weight,bias=None,stride=1,padding=(2,0), dilation=2)
    data_grad = data_grad.view(init_shape)

    sign_data_grad = data_grad.sign()
	
    perturbed_image = tensor_image + epsilon*sign_data_grad
    return data_grad * epsilon

class SMIA(object):
    def __init__(self, model = None, epsilon = None, loss_fn = None):
        """
        STABILIZED MEDICAL IMAGE ATTACKS
	a1, a2: balancing the influence of LDEV and LSTA
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = loss_fn

    def perturb(self, X, y, a1 = 1, a2 = 0, epsilons=None, niters=None):
        """
        Given examples (X, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """
        # Providing epsilons in batch
        if epsilons is not None:
            self.epsilon = epsilons

        use_cuda = torch.cuda.is_available()
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
  
        X_pert = Variable(torch.tensor(X.clone()), volatile=True).cuda()
        X_pert.requires_grad = True

        for i in range(niters):
            output_perturbed = None
            output_perturbed = self.model(X_pert.cuda())
            if i == 0:
                y = (y>0).to(torch.long)
                loss = self.loss_fn(output_perturbed, y)
            else:
                if len(gt1.shape) == 4:
                    label = gt1.squeeze(0)
                else:
                    label = gt1.squeeze(0).squeeze(0)
                # label = (label>0).to(torch.long)
                loss = a1 * self.loss_fn(output_perturbed, label.to(torch.long)) 
                loss = loss - a2 * self.loss_fn(output_perturbed, output_perturbed_last)
            loss.backward()
            X_pert_grad=X_pert.grad.detach().sign()
            pert = X_pert.grad.detach().sign() * self.epsilon
            
            kernel = gkern(1, 1).astype(np.float32)
            stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
            stack_kernel = np.expand_dims(stack_kernel, 3)

            gt1=X_pert_grad.detach()
            gt1=blur(gt1, self.epsilon, stack_kernel)
            gt1=Variable(torch.tensor(X_pert + gt1), volatile=True).cuda()
            gt1.requires_grad_(False)
            output_perturbed_last = self.model(gt1)
            _, output_perturbed_last = torch.max(output_perturbed_last, dim=1)
        
            X_pert = torch.clamp(Variable(torch.tensor(X_pert), volatile=True).cuda()+ pert.cuda(), min=0)
            X_pert.requires_grad = True
      
        return X_pert


