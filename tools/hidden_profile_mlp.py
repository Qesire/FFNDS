"""One fixed cross-coordinate C-profile head, distinct from the rejected HP1."""
import torch
import torch.nn.functional as F


def train(hidden, profiles, seed):
    x, targets = hidden.float(), profiles.float() * 216
    mean = x.mean(0)
    scale = (x-mean).square().mean(0).sqrt().clamp_min(1e-4)
    x = (x-mean)/scale
    with torch.random.fork_rng(devices=[hidden.device.index]):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(torch.nn.Linear(256,128), torch.nn.ReLU(), torch.nn.Linear(128,216)).to(hidden.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=.01)
    generator = torch.Generator(device=hidden.device).manual_seed(seed+1)
    losses, updates = [], 0
    for epoch in range(20):
        order = torch.randperm(len(x), generator=generator, device=x.device)
        total = 0.
        for ids in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x[ids]), targets[ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            updates += 1
        losses.append(total/len(x))
    if updates != 240:
        raise RuntimeError('Exactly240 updates required')
    with torch.no_grad():
        w1 = model[0].weight / scale
        b1 = model[0].bias - w1 @ mean
        parameters = {'w1':w1.contiguous(), 'b1':b1.contiguous(),
                      'w2':model[2].weight.detach().contiguous(), 'b2':model[2].bias.detach().contiguous()}
    return parameters, model, mean, scale, {'epoch_training_MSE':losses, 'updates':updates}


def predict(hidden, parameters):
    value = F.relu(F.linear(hidden.float(), parameters['w1'], parameters['b1']))
    return F.linear(value, parameters['w2'], parameters['b2'])
