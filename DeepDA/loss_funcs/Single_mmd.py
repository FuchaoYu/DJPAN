import torch
import torch.nn as nn

class Single_MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(Single_MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None        # 是否固定，如果固定，则为单核MMD
        self.kernel_type = kernel_type

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
        total = torch.cat([source, target], dim=0)      # tensor拼接
        total0 = total.unsqueeze(0).expand(             # 升维
            int(total.size(0)), int(total.size(0)), int(total.size(1)))  # 对总样本变换格式为（b1+b2,1,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按行复制
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))  # 对总样本变换格式为（b1+b2,1,n）,然后将后两维度数据复制到新拓展的维度上（b1+b2，b1+b2,n），相当于按列复制
        L2_distance = ((total0-total1)**2).sum(2)
        # 计算高斯核中的|x-y| 欧式距离。求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
        # 注意！！！！！ 在该部分计算中，已经完成了对每个坐标点欧氏距离的平方后求和（D({x1,.....x2},{y1,...y2})=D(x1,y1)+...+D(xn,yn)），即不需要再求矩阵M并与其相乘
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            ebandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)  # L2/N(N-1)
            # bandwidth = 2*1**2
        kernel_val = torch.exp(-L2_distance / bandwidth)
        return kernel_val

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
