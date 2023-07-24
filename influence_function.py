#! /usr/bin/env python3

import torch
from torch.autograd import grad
from pytorch_influence_functions.utils import display_progress
import torch.nn.functional as F


def s_test(X_test_sample, y_test_sample, w, X_train, y_train, gpu=-1, damp=0.01, scale=25.0,
           recursion_depth=5000):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""
    v = grad_z(X_test_sample, y_test_sample, w, gpu)
    h_estimate = v.copy()

    ################################
    # TODO: Dynamically set the recursion depth so that iterations stops
    # once h_estimate stabilises
    ################################
    for i in range(recursion_depth):
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once, break at the end of loop
        #########################
        # TODO: do x, t really have to be chosen RANDOMLY from the train set?
        #########################
        for i in range(len(X_train)):
            x = X_train[i]
            t = y_train[i]
        #for _, x, t, _, _ in z_loader:
            if gpu >= 0:
                x, t = x.cuda(), t.cuda()
            #y = model(x)[0]
            #y = model(x)
            
            y = torch.matmul(z, w)
    
            loss = lr_loss(w, z, t, 1e-6)
            #params = [ p for p in model.parameters() if p.requires_grad ]
            hv = hvp(loss, w, h_estimate)
            # Recursively caclulate h_estimate
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            break
        display_progress("Calc. s_test recursions: ", i, recursion_depth)
    return h_estimate


def calc_loss(y, t):
    """Calculates the loss

    Arguments:
        y: torch tensor, input with size (minibatch, nr_of_classes)
        t: torch tensor, target expected by loss of size (0 to nr_of_classes-1)

    Returns:
        loss: scalar, the loss"""
    ####################
    # if dim == [0, 1, 3] then dim=0; else dim=1
    ####################
    # y = torch.nn.functional.log_softmax(y, dim=0)
    y = torch.nn.functional.log_softmax(y)
    t = t.squeeze(1)
    #print(y.size())
    #print(t.size())
    loss = torch.nn.functional.nll_loss(
        y, t, weight=None, reduction='mean')
    return loss




def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True, allow_unused=True)
    


    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads[:1], v[:1]):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True, allow_unused=True)[:1]

    return return_grads

def lr_loss(w, X, y, lam):
    return -F.logsigmoid(y * torch.matmul(X, w)).mean() + lam * w.pow(2).sum() / 2

def grad_z(z, t, w, gpu=-1):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    # initialize
    if gpu >= 0:
        z, t = z.cuda(), t.cuda()
        
    #y = model(z)[0]
    #y = model(z)
    y = torch.matmul(z, w)
    
    loss = lr_loss(w, z, t, 1e-6)
    # Compute sum of gradients from model parameters to loss

    return list(grad(loss, w, create_graph=True, allow_unused=True))