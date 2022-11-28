# """Pytorch implementation of Class-Balanced-Loss
#    Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
#    Authors: Yin Cui and
#                Menglin Jia and
#                Tsung Yi Lin and
#                Yang Song and
#                Serge J. Belongie
#    https://arxiv.org/abs/1901.05555, CVPR'19.
# """


# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()
        

# # def focal_loss(labels, logits, alpha, gamma):
# #     """Compute the focal loss between `logits` and the ground truth `labels`.

# #     Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
# #     where pt is the probability of being classified to the true class.
# #     pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

# #     Args:
# #       labels: A float tensor of size [batch, num_classes].
# #       logits: A float tensor of size [batch, num_classes].
# #       alpha: A float tensor of size [batch_size]
# #         specifying per-example weight for balanced cross entropy.
# #       gamma: A float scalar modulating loss from hard and easy examples.

# #     Returns:
# #       focal_loss: A float32 scalar representing normalized total loss.
# #     """    
# #     BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels, reduction = "none")

# #     if gamma == 0.0:
# #         modulator = 1.0
# #     else:
# #         modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

# #     loss = modulator * BCLoss

# #     weighted_loss = alpha * loss
# #     focal_loss = torch.sum(weighted_loss)

# #     focal_loss /= torch.sum(labels)
# #     return focal_loss



# # def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
# #     """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

# #     Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
# #     where Loss is one of the standard losses used for Neural Networks.

# #     Args:
# #       labels: A int tensor of size [batch].
# #       logits: A float tensor of size [batch, no_of_classes].
# #       samples_per_cls: A python list of size [no_of_classes].
# #       no_of_classes: total number of classes. int
# #       loss_type: string. One of "sigmoid", "focal", "softmax".
# #       beta: float. Hyperparameter for Class balanced loss.
# #       gamma: float. Hyperparameter for Focal loss.

# #     Returns:
# #       cb_loss: A float tensor representing class balanced loss
# #     """
# #     effective_num = 1.0 - np.power(beta, samples_per_cls)
# #     weights = (1.0 - beta) / np.array(effective_num)
# #     weights = weights / np.sum(weights) * no_of_classes

# #     labels_one_hot = F.one_hot(labels, no_of_classes).float()

# #     weights = torch.tensor(weights).float()
# #     weights = weights.unsqueeze(0)
# #     weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
# #     weights = weights.sum(1)
# #     weights = weights.unsqueeze(1)
# #     weights = weights.repeat(1,no_of_classes)

# #     if loss_type == "focal":
# #         cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
# #     elif loss_type == "sigmoid":
# #         cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
# #     elif loss_type == "softmax":
# #         pred = logits.softmax(dim = 1)
# #         cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
# #     return cb_loss



# # if __name__ == '__main__':
# #     no_of_classes = 5
# #     logits = torch.rand(10,no_of_classes).float()
# #     labels = torch.randint(0,no_of_classes, size = (10,))
# #     beta = 0.9999
# #     gamma = 2.0
# #     samples_per_cls = [2,3,1,2,2]
# #     loss_type = "focal"
# #     cb_loss = CB_loss(labels, logits, samples_per_cls, no_of_classes,loss_type, beta, gamma)
# #     print(cb_loss)






import numpy as np
import torch
import torch.nn.functional as F


def focal_loss(logits, labels, alpha=None, gamma=2):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bc_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bc_loss

    if alpha is not None:
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
    else:
        focal_loss = torch.sum(loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class CBLoss(torch.nn.Module):
    def __init__(
        self,
        loss_type: str = "cross_entropy",
        beta: float = 0.999,
        fl_gamma=2,
        samples_per_class=None,
        class_balanced=False,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        reference: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
        Args:
            loss_type: string. One of "focal_loss", "cross_entropy",
                "binary_cross_entropy", "softmax_binary_cross_entropy".
            beta: float. Hyperparameter for Class balanced loss.
            fl_gamma: float. Hyperparameter for Focal loss.
            samples_per_class: A python list of size [num_classes].
                Required if class_balance is True.
            class_balanced: bool. Whether to use class balanced loss.
        Returns:
            Loss instance
        """
        super(CBLoss, self).__init__()

        if class_balanced is True and samples_per_class is None:
            raise ValueError("samples_per_class cannot be None when class_balanced is True")

        self.loss_type = loss_type
        self.beta = beta
        self.fl_gamma = fl_gamma
        self.samples_per_class = samples_per_class
        self.class_balanced = class_balanced

    def forward(
        self,
        logits: torch.tensor,
        labels: torch.tensor,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
            logits: A float tensor of size [batch, num_classes].
            labels: An int tensor of size [batch].
        Returns:
            cb_loss: A float tensor representing class balanced loss
        """

        batch_size = logits.size(0)
        num_classes = logits.size(1)
        # labels_one_hot = F.one_hot(labels, num_classes).float()
        labels_one_hot = labels.float()

        if self.class_balanced:
            effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
            weights = (1.0 - self.beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * num_classes
            weights = torch.tensor(weights, device=logits.device).float()

            if self.loss_type != "cross_entropy":
                weights = weights.unsqueeze(0)
                weights = weights.repeat(batch_size, 1) * labels_one_hot
                weights = weights.sum(1)
                weights = weights.unsqueeze(1)
                weights = weights.repeat(1, num_classes)
        else:
            weights = None

        if self.loss_type == "focal_loss":
            cb_loss = focal_loss(logits, labels_one_hot, alpha=weights, gamma=self.fl_gamma)
        elif self.loss_type == "cross_entropy":
            cb_loss = F.cross_entropy(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "binary_cross_entropy":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
        elif self.loss_type == "softmax_binary_cross_entropy":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss