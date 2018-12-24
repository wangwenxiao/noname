import torch

def L2Norm(r):
    return (r**2).sum(1).sqrt().view(r.size(0), 1)

def Normalize(r):
    return r / L2Norm(r)

def L2_normalized_distance(r1, r2):
    r1 = r1.view(r1.size(0), -1)
    r2 = r2.view(r2.size(0), -1)
    
    r1 = Normalize(r1)
    r2 = Normalize(r2)
    
    return L2Norm(r1 - r2).mean().item()
