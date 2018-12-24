import torch

def BinaryIOU(r1, r2):
    r1 = r1.view(r1.size(0), -1)
    r2 = r2.view(r2.size(0), -1)
    
    r1 = r1.sign()
    r2 = r2.sign()

    N = r1.size(1)
    
    I = ((r1 * r2).sum(1) + N) / 2
    U =  N * 2 - I
    return (I / U).mean().item()

def BinaryION(r1, r2):
    r1 = r1.view(r1.size(0), -1)
    r2 = r2.view(r2.size(0), -1)
    
    r1 = r1.sign()
    r2 = r2.sign()

    N = r1.size(1)
    
    I = ((r1 * r2).sum(1) + N) / 2
    return (I / N).mean().item()
