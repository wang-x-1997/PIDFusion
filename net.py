import torch
from torch import nn
import torch.nn.functional as F

from TD import TD
import numpy as np

def Sobel(image, channel,cuda_visible=True):
    assert torch.is_tensor(image) is True
    # [-1.,-2.,-1.], [0.,0.,0.,], [1.,2.,1.]
    # [0., 1., 0.], [1., -4., 1.], [0., 1., 0.]
    # [-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1]
    laplace_operator = np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=np.float32)[np.newaxis,
                       :, :].repeat(channel, 0)
    if cuda_visible:
        laplace_operator = torch.from_numpy(laplace_operator).unsqueeze(0).cuda()
    else:
        laplace_operator = torch.from_numpy(laplace_operator).unsqueeze(0)

    image = F.conv2d(image, laplace_operator,padding=1, stride=1)

    return image
class GenNoise(nn.Module):
    def __init__(self, channels=64, num_of_layers=2):
        super(GenNoise, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding))
        layers.append(nn.BatchNorm2d(features))
        layers.append(nn.ReLU())
        for _ in range(num_of_layers-1):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=1, padding=0))
        self.genclean = nn.Sequential(*layers)
        # for m in self.genclean:
        #     if isinstance(m, nn.Conv2d):
        #        nn.init.xavier_uniform(m.weight)
        #        nn.init.constant(m.bias, 0)

    def forward(self, x):
        out = self.genclean(x)
        return out


class VisualPIDNet(nn.Module):
    def __init__(self, channel=32):
        super(VisualPIDNet, self).__init__()
        self.channel = channel
        self.p_branch = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.i_fc = nn.Linear(channel, channel)
        self.d_fc = nn.Linear(channel, channel)
        self.feature_to_image = nn.Conv2d(channel, channel, kernel_size=1)
        self.integral = None
        self.prev_error = None

    def forward(self, x):

        p_out = self.p_branch(x)


        if self.integral is None:
            self.integral = torch.zeros_like(F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)).to(x.device)

        self.integral = torch.clamp(self.integral + F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1).clone().detach(), min=-10, max=10)

        global_features = self.integral
        i_out = self.i_fc(global_features).unsqueeze(-1).unsqueeze(-1)

        if self.prev_error is None:
            self.prev_error = torch.zeros_like(global_features).to(global_features.device)
        delta_error = global_features - self.prev_error.clone().detach()
        self.prev_error = global_features.clone()
        d_out = self.d_fc(delta_error).unsqueeze(-1).unsqueeze(-1)

        combined_features = p_out + i_out + d_out

        return combined_features

class VisualLSTM(nn.Module):
    def __init__(self,C=32):
        super(VisualLSTM, self).__init__()
        self.hidden_size = C
        self.num_layers = 1
        self.lstm = nn.LSTM(C, self.hidden_size, 1, batch_first=True)
        # self.mlp =
        self.FEA =  FeatureEnhancementWithAttention(32)

    def forward(self, source,hn=1, cn=1, item = True):

        B,C,H,W = source.shape
        x = F.adaptive_avg_pool2d(source,1)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BLC

        if item:
            h0 = torch.zeros(1, B, self.hidden_size).to(source.device)
            c0 = torch.zeros(1, B, self.hidden_size).to(source.device)
        else:
            h0 = hn
            c0 = cn

        output, (hn, cn) = self.lstm(x, (h0, c0))
        output = output.reshape(B, C, -1)  # B*C*H*W -> B*C*1

        return self.FEA(source,output), (hn, cn)


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel):
        super(ChannelAttentionModule, self).__init__()
        self.fc1 = nn.Linear(channel, channel // 16)
        self.fc2 = nn.Linear(channel // 16, channel)

    def forward(self, x):
        # x: (B, C, 1)
        x = torch.squeeze(x, -1)  # (B, C)
        x = F.relu(self.fc1(x))  # (B, C // 16)
        x = torch.sigmoid(self.fc2(x))  # (B, C), 注意力权重
        return x


class FeatureEnhancementWithAttention(nn.Module):
    def __init__(self, channel):
        super(FeatureEnhancementWithAttention, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)

    def forward(self, original_feature, representation_vector):
        # original_feature: (B, C, H, W)
        # representation_vector: (B, C, 1)

        attention_weights = self.channel_attention(representation_vector)  # (B, C)
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        enhanced_feature = original_feature * attention_weights  # (B, C, H, W)

        return enhanced_feature
class Tex(nn.Module):
    def __init__(self,channels=32):
        super(Tex, self).__init__()
        self.con = nn.Conv2d(1, channels, 1, 1, 0)
        self.channel = channels
    def forward(self,x):
        T = self.con(Sobel(x, channel=self.channel))
        return T

class Net(nn.Module):
    def __init__(self,N=2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,1,1)

        self.N = N
        self.PID = VisualPIDNet()
        self.Noise = GenNoise(channels=32, num_of_layers=1)
        self.LSTM = VisualLSTM()

        self.decoder = nn.Sequential(
            # nn.Conv2d(128, 64, 3, 1, 1),
            # # nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, 1, 1),
            # nn.Sigmoid()
        )
        self.tex = Tex(channels=32)


    def forward(self,IR,VIS):
        if self.training:
            fea_Noise_list =[]
            fea_clean_list = []
            fea_source_list = []

        fea_M = self.conv1(IR-VIS)

        # fea_LSTM,hidden = self.LSTM(fea_M)
        fea_PID = self.PID(fea_M )
        fea_tex = self.tex(fea_M)
        fea_Noise = self.Noise(fea_PID + fea_tex)

        fea_1 = fea_PID - fea_Noise + fea_tex
        if self.training:
            fea_Noise_list.append(fea_Noise)
            fea_source_list.append(fea_M)
            fea_clean_list.append(fea_1)

        for i in range(self.N):
            fea_LSTM,hidden = self.LSTM(fea_1)
            fea_LSTM +=fea_M
            fea_tex = self.tex(fea_LSTM)
            fea_PID = self.PID(fea_LSTM)
            fea_Noise = self.Noise(fea_PID +fea_tex)
            fea_1 = fea_PID +fea_tex- fea_Noise
            if self.training:
                fea_Noise_list.append(fea_Noise)
                fea_source_list.append(fea_LSTM)
                fea_clean_list.append(fea_1)
        fea_LSTM, hidden = self.LSTM(fea_1 )
        fea_LSTM += fea_M
        Y_final = norm_1(self.decoder(fea_LSTM))

        if self.training:
            return (VIS+Y_final),self.Noise_loss(fea_source_list ,fea_clean_list,fea_Noise_list,Y_final,IR-VIS)
        else:
            return (VIS + Y_final),(Y_final)

    def Noise_loss(self,fea_source_list,fea_clean_list,fea_Noise_list,Y_final,M ):
        loss1 = 0
        loss2 = 0
        loss3 = 0
        loss4 = 0
        loss5 = torch.norm(Y_final-salency(M),p=1)
        for i in range(len(fea_source_list)):
            loss1 += torch.norm(fea_source_list[i]-fea_Noise_list[i]-fea_clean_list[i])
            fea_Noise_list_1 = self.Noise(fea_Noise_list[i])
            fea_clean_list_1 = self.Noise(fea_clean_list[i])
            loss2 += torch.norm(fea_Noise_list_1 - fea_Noise_list[i],p=1)
            loss3 += torch.norm(fea_clean_list_1-torch.zeros_like(fea_source_list[i]),p=1)
            loss4 += torch.norm(fea_clean_list[i]-salency(fea_clean_list[i]),p=1)
        return (loss1+loss2+loss3+loss4)/32/16



def salency(x):
    return norm_1((x-torch.mean(x)))
def norm_1(x):
    max1 = torch.max(x)
    min1 = torch.min(x)
    norm = (x - min1) / (max1 - min1 + 1e-10)
    # return (norm-0.5)*2
    return norm