import numpy as np
import torch
import csv
import pandas as pd
import random
import matplotlib.pyplot as plt
from torch import nn, optim, autograd
from torch.autograd import Variable
from load_data import load_wind, data_generator
from model import Generator, Discriminator, gradient_penalty
from visdom import Visdom

viz = Visdom()
viz.line([[0.0, 0.0]], [0.], win='loss', opts=dict(title='loss', legend=['D_loss', 'G_loss']))

torch.set_default_tensor_type(torch.cuda.FloatTensor)

trX, trY, mean, std = load_wind()

D = Discriminator().cuda()
G = Generator().cuda()

print(D)
data_iter = data_generator(trX, trY)

optim_G = optim.RMSprop(G.parameters(), lr = 1e-4)
optim_D = optim.RMSprop(D.parameters(), lr = 1e-4)

g_loss = []
d_loss = []
ep = []
p_real = []
p_fake = []


for epoch in range(40000):

    for _ in range(5):
        xr, yr = next(data_iter)
        xr = torch.from_numpy(xr).cuda()  # (32, 1, 24, 24)
        yr = torch.from_numpy(yr).cuda()  # (32, 5)

        # xr.requires_grad_()
        # yr = torch.tensor(yr, dtype=torch.float)
        predr = D(xr, yr)  # (32, 1)
        lossr = -predr.mean()

        zx = torch.randn((32, 100), dtype=torch.float).cuda()
        zx = torch.cat([zx, yr], dim=1)  # (32, 105)
        # zx = torch.tensor(zx, dtype=torch.float)
        xf = G(zx).detach()  # xf (32, 1, 24, 24)
        predf = D(xf, yr)
        lossf = predf.mean()

        gp = gradient_penalty(D, xr, xf.detach(), yr)
        loss_D = lossr + lossf + 10 * gp

        optim_D.zero_grad()
        loss_D.backward()
        optim_D.step()

    z = torch.randn((32, 100), dtype=torch.float).cuda()
    x, y = next(data_iter)
    # y = torch.tensor(y, dtype=torch.float)
    y = torch.from_numpy(y).cuda()
    zx = torch.cat([z, y], dim=1)
    # zx = torch.tensor(zx, dtype=torch.float)
    xf = G(zx)

    predf = D(xf, y)
    loss_G = -predf.mean()

    optim_G.zero_grad()
    loss_G.backward()
    optim_G.step()

    g_loss.append(loss_G.item())
    d_loss.append(loss_D.item())
    ep.append(epoch)

    xp, yp = next(data_iter)
    xp = torch.from_numpy(xp).cuda()
    yp = torch.from_numpy(yp).cuda()
    zp = torch.randn(32, 100).cuda()

    prer = D(xp, yp)
    p_r = prer.mean()
    p_real.append(p_r.item())
    zp = torch.cat([zp, yp], dim=1)
    xff = G(zp)
    pref = D(xff, yp)
    p_f = pref.mean()
    # print('aa',p_f)
    # print(type(p_f))
    # print('bbb', p_f.item())
    p_fake.append(p_f.item())

    if epoch % 100 == 0:
        print('epoch', epoch)
        print(loss_D.item(), loss_G.item())

        viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')
        torch.save(G.state_dict(), './rmsprop/generator.pth')
        torch.save(D.state_dict(), './rmsprop/discriminator.pth')


plt.plot(ep, g_loss, label='gen_loss')
plt.xlabel('epoch')
plt.ylabel('gen_loss')
plt.title('gen_loss')
plt.savefig('./rmsprop/gen_loss.jpg')
plt.show()
plt.close()

plt.plot(ep, d_loss, label='discri_loss')
plt.xlabel('epoch')
plt.ylabel('discri_loss')
plt.title('discri_loss')
plt.savefig('./rmsprop/discri_loss.jpg')
plt.show()
plt.close()


real = []
fake = []
xla = []
G.eval()
for i in range(30):
    p, q = next(data_iter)
    p = torch.from_numpy(p).cuda()
    q = torch.from_numpy(q).cuda()
    rp = p.mean()
    # torch.manual_seed(0)
    fp = torch.randn(32, 100).cuda()
    fp = torch.cat([fp, q], dim=1)
    rrp = fp.mean()
    real.append(rp.item())
    fake.append(rrp.item())
    xla.append(i)


plt.plot(xla, real)
plt.savefig('./rmsprop/1real.jpg')
plt.show()
plt.close()

plt.plot(xla, fake)
plt.savefig('./rmsprop/1fake.jpg')
plt.show()
plt.close()



