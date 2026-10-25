"""Real tanh-GELU derivative and a conditional matrix-folding cost identity."""
import math


def derivative(a):
    import torch
    k=math.sqrt(2/math.pi)
    t=torch.tanh(k*(a+.044715*a.pow(3)))
    return .5*(1+t)+.5*a*(1-t.square())*k*(1+3*.044715*a.square())


def folding_mac_ratio(groups, m=31668, d=5120, h=13824):
    # groups D×H by H×D compositions, plus M×D by D×D application.
    return groups*d/(2*m)+d/(2*h)
