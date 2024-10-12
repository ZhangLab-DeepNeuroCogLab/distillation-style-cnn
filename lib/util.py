import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from PIL import Image
import torch.distributed as dist
from torchvision.transforms import transforms
import logging

np.random.seed(0)

class MixDataLoader(object):
    r"""An abstract class representing a :class:`Dataset`.
    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.
    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs a index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __init__(self, dataloader_main, dataloader_aux):
        self.dataloader_main = dataloader_main
        self.dataloader_aux = dataloader_aux
        assert len(dataloader_main) == len(dataloader_aux)
        self.len = len(dataloader_main)

    def __len__(self):
        return self.len

    def __iter__(self):
        self.dataloader_main_iter = self.dataloader_main.__iter__()
        self.dataloader_aux_iter = self.dataloader_aux.__iter__()
        return self

    def __next__(self):
        inputs_main, target_main = next(self.dataloader_main_iter)
        inputs_aux, target_aux = next(self.dataloader_aux_iter)
        return (torch.cat([inputs_main, inputs_aux]), torch.cat([target_main, target_aux]))

def cutout(mask_size=16, p=1, cutout_inside=True, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout

class GaussianNoise(object):
    def __init__(self, mean=0., std=0.025):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

def probabilities_to_decision(probabilities,cls_names, aggregation_function=np.mean):
        """Return one of 16 categories for vector of probabilities.

        Keyword arguments:
        probabilities -- a np.ndarray of length 1000
                         (softmax output: all values should be
                         within [0,1])
        """

        assert type(probabilities) is np.ndarray
        assert (probabilities >= 0.0).all() and (probabilities <= 1.0).all()
        assert len(probabilities) == 100

        cat =      ['cat','n02123159','n02124075']
        boat =     ['boat','n02951358']
        clock =    ['clock','n04548280']
        keyboard = ['keyboard','n03085013']
        truck =    ['truck','n03417042','n03977966']
        bird =     ['bird','n01558993', 'n01560419', 'n01592084',
                'n01795545', 'n01829413', 'n01855672', 'n02017213']
        dog =      ['dog','n02088238', 'n02091467','n02093256','n02097298',
                    'n02100236', 'n02106030','n02106166','n02107312',
                     'n02107683','n02110063','n02110627','n02110958','n02113624']

        max_value = -float("inf")
        category_decision = None
        for categores in [cat,clock,dog,keyboard,truck,bird,boat]:
            indices = []
            for category in categores[1:]:
                indices.append(cls_names.index(category))
            values = np.take(probabilities, indices)
            aggregated_value = aggregation_function(values)
            if aggregated_value > max_value:
                max_value = aggregated_value
                category_decision = categores[0]
   
        return category_decision

class TransformTwice:
    def __init__(self, transform, aug_transform):
        self.transform = transform
        self.aug_transform = aug_transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.aug_transform(inp)
        return out1, out2

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.5):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.detach().data)


'''
def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        #p2.data.mul_(m).add_(1 - m, p1.detach().data)
        #p2.data.mul_(m).add_(p1.detach().data, 1-m)
        #p2.data = p2.data*m + p1.detach().data*(1-m)
        #p2.data.mul_(m).add_(1. - m, p1.data)
        p2.data.copy_(p1.data)
        p2.requires_grad = False
'''

def set_bn_train(model):
    def set_bn_train_helper(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()

    model.eval()
    model.apply(set_bn_train_helper)


def moment_update(model, model_ema, m, global_step):
    """ model_ema = m * model_ema + (1 - m) model """
    m = min(1 - 1 / (global_step + 1), m)
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)

def weight_norm(model):
    w_data_a = model.fc.fc1.weight.data
    w_data_b = model.fc.fc2.weight.data

    L_norm_a = torch.norm(w_data_a, p=2, dim =1)
    L_norm_b = torch.norm(w_data_b, p=2, dim =1)
    #print ('norm_a:', L_norm_a, ' norm_b: ', L_norm_b)
    # print ('norm_a mean: ', L_norm_a.mean(0), ' norm_/b norm: ', L_norm_b.mean(0))
    logging.info(f'norm_a mean: {L_norm_a.mean(0)}, norm_/b norm:{L_norm_b.mean(0)}')
    return L_norm_a.mean(0)/L_norm_b.mean(0)

def weight_norm_dot(model):
    w_data_a = model.fc.weight.data
    L_norm_a = torch.norm(w_data_a, p=2, dim =1)
    #print ('norm_a:', L_norm_a)
    # print ('norm_old mean: ', L_norm_a[:-10].mean(0), 'norm_new mean: ', L_norm_a[-10:].mean(0))
    logging.info(f'norm_old mean: {L_norm_a[:-10].mean(0)}, norm_new mean:{L_norm_a[-10:].mean(0)}')
    return L_norm_a[:-10].mean(0)/L_norm_a[-10:].mean(0)
    #return L_norm_a

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Do not log KeyboardInterrupt to avoid cluttering the log with user-initiated interrupts
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Log the uncaught exception
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))