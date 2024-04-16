import torch
import torch.nn as nn
import numpy as np


class djp_mmdloss(nn.Module):
    def __init__(self, num_class, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(djp_mmdloss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None  # 是否固定，如果固定，则为单核MMD
        self.kernel_type = kernel_type
        self.num_class = num_class

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        '''
            多核或单核高斯核矩阵函数，根据输入样本集x和y，计算返回对应的高斯核矩阵
            Params:
             source: (b1,n)的X分布样本数组
             target:（b2，n)的Y分布样本数组
             kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
             kernel_num: 取不同高斯核的数量
             fix_sigma: 是否固定，如果固定，则为单核MMD
            Return:
              sum(kernel_val): 多个核矩阵之和
        '''

        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)  # tensor拼接
        total0 = total.unsqueeze(0).expand(  # 升维
            int(total.size(0)), int(total.size(0)),
            int(total.size(1)))  # 对总样本变换格式为（b1+b2,1,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按行复制
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)),
            int(total.size(1)))  # 对总样本变换格式为（b1+b2,1,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按列复制
        L2_distance = ((total0 - total1) ** 2).sum(2)
        # 计算高斯核中的|x-y| 欧式距离。求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
        # 注意！！！！！ 在该部分计算中，已经完成了对每个坐标点欧氏距离的平方后求和（D({x1,.....x2},{y1,...y2})=D(x1,y1)+...+D(xn,yn)），即不需要再求矩阵M并与其相乘
        if fix_sigma:
            bandwidth = 2 * fix_sigma ** 2
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)  # L2/N(N-1)
        # 下面开始进行多核计算
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)  # 多核中的β值为 kernel_mul**（i-kernel_num // 2）
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      # 前面根据mul与num得到了5个bandwidth值，此时便有5个kernel。或者理解为σ取多个值，分别求核函数然后取和，作为最后的核函数。
                      for bandwidth_temp in bandwidth_list]  # 所谓的多核，即σ不同
        return sum(kernel_val)  # 多个核矩阵之和

    def forward(self, source, target, source_label, target_logits):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            m = batch_size
            num_class = self.num_class
            C = num_class
            n = int(target.size()[0])
            Ys = torch.zeros(m, C).scatter_(1, source_label.unsqueeze(1).cpu(), 1)
            target_label = target_logits.cpu().data.max(1)[1].numpy()
            Yt = np.eye(self.num_class)[target_label]
            Yt = torch.FloatTensor(Yt)


            Rmin_1 = torch.cat((torch.mm(Ys, Ys.T), torch.mm(Ys, Yt.T)), 0)
            Rmin_2 = torch.cat((torch.mm(Yt, Ys.T), torch.mm(Yt, Yt.T)), 0)
            Rmin = torch.cat((Rmin_1, Rmin_2), 1)

            Ms = torch.empty(m, (C - 1) * C)
            Mt = torch.empty(n, (C - 1) * C)
            for i in range(0, C):
                idx = torch.arange((C - 1) * i, (C - 1) * (i + 1))
                Ms[:, idx] = Ys[:, i].repeat(C - 1, 1).T
                tmp = torch.arange(0, C)
                Mt[:, idx] = Yt[:, tmp[tmp != i]]
            Rmax_1 = torch.cat((torch.mm(Ms, Ms.T), torch.mm(Ms, Mt.T)), 0)
            Rmax_2 = torch.cat((torch.mm(Mt, Ms.T), torch.mm(Mt, Mt.T)), 0)
            Rmax = torch.cat((Rmax_1, Rmax_2), 1)

            M = Rmin - 0.1 * Rmax

            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            temp = torch.mm(kernels.cpu(),M)
            XX = torch.mean(temp[:batch_size, :batch_size])
            YY = torch.mean(temp[batch_size:, batch_size:])
            XY = torch.mean(temp[:batch_size, batch_size:])
            YX = torch.mean(temp[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            loss = torch.abs(loss)
            return loss
