"""Per-channel calibration and exact arithmetic accounting for affine folding."""


def fit_coefficients(preactivation,labels):
    a=preactivation-preactivation.mean(0,keepdim=True)
    q=labels-labels.mean(0,keepdim=True)
    slope=(a*q).sum(0,keepdim=True)/a.square().sum(0,keepdim=True).clamp_min(1e-30)
    return slope,labels.mean(0,keepdim=True)-slope*preactivation.mean(0,keepdim=True)


def absolute_fold_mac_ratio(corrected_groups):
    # D=80*64, H=216*64. The D×D map costs exactly40 of216
    # original FFN groups, counting its two D×H/H×D projections.
    return (40+corrected_groups)/216
