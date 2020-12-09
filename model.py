import torch.nn as nn
import torch


def weights_init(m):
    """
    权重初始化
    :param m:
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class MLP_G(nn.Module):
    """"""
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, noise, att):
        """
        :param noise: 噪声
        :param att: 属性
        :return:
        """
        h = torch.cat((noise, att), 1) # Att加入噪声
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

class Mapping(nn.Module):
    def __init__(self, opt):
        super(Mapping, self).__init__()
        self.latensize=opt.latenSize
        self.encoder_linear = nn.Linear(opt.resSize, opt.latenSize*2)
        self.discriminator = nn.Linear(opt.latenSize, 1) # 全连接层输出到一个Scalar 作为判别
        self.classifier = nn.Linear(opt.latenSize, opt.nclass_seen) # 全连接层Encode到已见类的数量
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init) # 权重初始化

    def forward(self, x, train_G=False):
        laten=self.lrelu(self.encoder_linear(x))
        mus,stds = laten[:,:self.latensize],laten[:,self.latensize:]
        stds=self.sigmoid(stds)
        encoder_out = reparameter(mus, stds)
        if not train_G:
            dis_out = self.discriminator(encoder_out)
        else:
            dis_out = self.discriminator(mus)
        pred=self.logic(self.classifier(mus))

        return mus,stds,dis_out,pred,encoder_out

class Latent_CLS(nn.Module):
    def __init__(self, opt):
        super(Latent_CLS, self).__init__()
        self.classifier = nn.Linear(opt.latenSize, opt.nclass_seen)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, latent):
        pred = self.logic(self.classifier(latent))
        return pred

class Encoder(nn.Module):
    """ """
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.latensize = opt.latenSize
        self.encoder_linear = nn.Linear(opt.resSize, opt.latenSize * 2)

        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)
    def forward(self, x,mean_mode=True):
        """
        :param x:
        :param mean_mode:
        :return:
        """
        latent = self.lrelu(self.encoder_linear(x))
        mus, stds = latent[:, :self.latensize], latent[:, self.latensize:]
        stds = self.sigmoid(stds)
        z=reparameter(mus, stds)
        return mus, stds,z

def reparameter(mu,sigma):
    return (torch.randn_like(mu) *sigma) + mu



"""
结构图中中间结果的Loss
"""
class D_middle(nn.Module):
    def __init__(self,X_dim,y_dim,h_dim):
        """
        中间Discriminator的初始化
        :param X_dim: 输入
        :param Y_dim: 输出
        :param h_dim: 隐藏
        """
        super(D_middle, self).__init__()

        self.D_shared = nn.Sequential(
            nn.Linear(X_dim,h_dim),
            nn.ReLU()
        )

        # branch one
        self.D_gan =nn.Linear(h_dim,1)

        # branch two
        self.D_aux = nn.Linear(h_dim,y_dim)
    def forward(self,input):
        h = self.D_shared(input)
        return self.D_gan(h),self.D_aux(h)









