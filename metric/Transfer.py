import torch

def L2Norm(r):
    return (r**2).sum(1).sqrt().view(r.size(0), 1)

def Normalize(r):
    return r / L2Norm(r)

def L2_mean(r1, r2):
    r1 = r1.view(r1.size(0), -1)
    r2 = r2.view(r2.size(0), -1)

    r1 = Normalize(r1)
    r2 = Normalize(r2)

    return (r1 * r2).sum(1).mean().item()

portion_base = 0

def setBase(x):
    global portion_base
    portion_base = x

def L2_portion(r1, r2):
    global portion_base
    r1 = r1.view(r1.size(0), -1)
    r2 = r2.view(r2.size(0), -1)

    r1 = Normalize(r1)
    r2 = Normalize(r2)

    d = (r1 * r2).sum(1)

    return float((d > portion_base).sum().item()) / d.size(0)

def Linf_mean(r1, r2):
    r1 = r1.view(r1.size(0), -1)
    r2 = r2.view(r2.size(0), -1)

    r1 = r1.sign()

    return ((r1 * r2).sum(1) / (r2.sign() * r2).sum(1)).mean().item()

def Linf_portion(r1, r2):
    global portion_base
    r1 = r1.view(r1.size(0), -1)
    r2 = r2.view(r2.size(0), -1)

    r1 = r1.sign()
    d = (r1 * r2).sum(1) / (r2.sign() * r2).sum(1)
    
    return float((d > portion_base).sum().item()) / d.size(0)

