# 파일 관리
import os
from glob import glob

# 이미지 처리
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from torchvision.utils import save_image

# For Custom Dataset
import torchvision.transforms.functional as TVF

# Network
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# For Custom Network
import math as m

# Utils
import time
from tqdm import tqdm

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class FFTBlock(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(FFTBlock, self).__init__()
        self.main_fft = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel*2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(out_channel*2, out_channel*2, kernel_size=1, stride=1)
        )
        self.norm = norm
    def forward(self, x):
        _, C, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        
        y = self.main_fft(y_f)
        
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return x + y    # 입력으로 들어온 값과, FFT층 통과한 이미지 더해서 return


# Channel 방향으로 쪼개서 Element-wise mul 진행
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # (batch, channel, row, col)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super(NAFBlock, self).__init__()
        dw_channel = c * DW_Expand  # Embedding 과정과, Non-linear 과정에서 정보유실을 방지하기위해 Expansion함 // 여기서는 SCA(채널쪼개서 서로 곱하는)부분에서 발생하는 non-linearity에서의 정보유실 방지
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        # GAP진행해서 Channel별로 정보 뽑고, Point-wise Conv 진행후에, 원래 feature와 channel-wise multiplication 진행
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        
        # 2@@@추가한부분, MLP인 1x1 point conv를 1d-conv로 교체했음. // 매우 빠르다는 장점도 있고, ECA에서 성능향상도 있었음. 기존의 SE Block을 대체하기 위해 나온 개념.
        #---------------------------------------------------#
        t = int(abs((m.log(dw_channel//2, 2) + 1) / 2))
        self.k_size = t if t % 2 else t + 1 # k_size가 홀수여야 padding이 가능함.
        self.sca1 = nn.AdaptiveAvgPool2d(1)
        self.sca2 = nn.Conv1d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size = self.k_size, padding=(self.k_size-1)//2, bias=False)   #@@@ 추가한 부분
        #---------------------------------------------------#

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c, eps=1e-6)
        self.norm2 = LayerNorm2d(c, eps=1e-6)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # Normalization 파라미터, 단순히 정규화해서 ReLU를 통과시키면, 대부분의 Param이 소멸되므로, beta와 같은 bias 추가
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)   # Expansion (point-wise conv)
        x = self.conv2(x)   # Depthwise Conv
        x = self.sg(x)      # 채널 쪼개서 곱하기 for non-linearity / 이 과정에서 Channel수가 반으로 줄음
        # x = x * self.sca(x) # Channel-wise Attension (element-wise mult)
        # 2@@@추가한부분, MLP인 1x1 point conv를 1d-conv로 교체했음. // 매우 빠르다는 장점도 있고, ECA에서 성능향상도 있었음. 기존의 SE Block을 대체하기 위해 나온 개념.
        #---------------------------------------------------#
        x = self.sca1(x)
        x = self.sca2(x.squeeze(-1)).unsqueeze(-1)  # conv1d는 [Batch_N, Channel, Length] 형식으로 받기때문에 기존의 [Batch_N, Channel, H, W]를 수정해줘야함
        #---------------------------------------------------#
        x = self.conv3(x)   # 원래 크기로 복구 (point-wise conv)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))   # 채널 수 ffn_channel으로 뻥튀기 (point-wise conv)
        x = self.sg(x)      # 채널 수 반(ffn_channel // 2)으로 줄음
        x = self.conv5(x)   # 원래 채널 수로 복구 (point-wise conv)

        x = self.dropout2(x)

        return y + x * self.gamma

class NAFNet(nn.Module):

    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)    # 최종적으로는 채널 3으로 나가야하니깐.

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        #------------------------------------------------------------------#
        self.fftBlock = nn.ModuleList() 
        #------------------------------------------------------------------#

        chan = width
        for num in enc_blk_nums:    # [1, 1, 1, 28]
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]   # 변수 num(1~28)이 여기에만 붙어있어. 즉 얘는 28번 반복. 하지만 down은 len(list) == 4번만 진행
                )
            )
            
            self.fftBlock.append(
                nn.Sequential(
                    FFTBlock(chan)
                )
            )
            self.downs.append(      # len(enc_blk_nums) == 4, 즉 2**4 배 만큼 이미지크기 감소
                nn.Conv2d(chan, 2*chan, kernel_size=2, stride=2)    # 이미지 크기는 절반 / 채널은 2 배
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan*2, kernel_size=1, bias=False),   # point-wise conv. 채널 수 2배 뻥튀기
                    
                    nn.PixelShuffle(upscale_factor=2)   # Feature map의 수많은 channel을 이용하여, pixel 위치에 맞는 각 Channel의 값을 떼어와서 feature map 확장  # 논문 참고 (https://mole-starseeker.tistory.com/m/84)
                                                        # 이미지를 가로 세로 2배씩 확장한다면, 채널은 4개가 필요함 // 따라서 채널이 upscale_factor ** 2 만큼 감소
                )
            )
            chan = chan // 2    # 채널 수 다시 2배 감소
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )


        self.padder_size = 2 ** len(self.encoders)  # == len(enc_blk_nums) == 4, down으로 인해 감소된 배수

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)    # padding 안함

        x = self.intro(inp)

        encs = []


        # [1, 1, 1, 28] 이 었으니, 4번만큼 반복
        for encoder, fftBlock, down in zip(self.encoders, self.fftBlock, self.downs):
            x = encoder(x)
            encs.append(fftBlock(x))
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)           # enc와 채널개수 및 이미지 크기를 맞춤
            x = x + enc_skip    # skip-connection 이용
            x = decoder(x)      # 

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class CustomRandomCrop(nn.Module):
    
    @staticmethod
    def get_params(img, output_size=(256, 256)):
        w, h = TVF._get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw

    def __init__(self, size):
        super().__init__()
        self.size = size

    def __call__(self, images):
        lq, gt = images['lq'], images['gt']

        width, height = TVF._get_image_size(gt)

        i, j, h, w = self.get_params(gt, self.size)

        return {'lq':TVF.crop(lq, i, j, h, w), 'gt':TVF.crop(gt, i, j, h, w)}

class CustomRandomHorizontalFlip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images):
        lq, gt = images['lq'], images['gt']
        if torch.rand(1) < self.p:
            return {'lq': TVF.hflip(lq), 'gt': TVF.hflip(gt)}
        
        return images

class CustomRandomRotation():
    def __init__(self, degrees=90, p=0.5):
        self.degrees = degrees
        self.p = p
  
    def __call__(self, images):
        lq, gt = images['lq'], images['gt']
        if torch.rand(1) < self.p:
            return {'lq': TVF.rotate(lq, angle=self.degrees), 'gt':TVF.rotate(gt, angle=self.degrees)}
        return images

class CustomToTensor():
    def __init__(self):
        pass
    def __call__(self, images):
        lq, gt = images['lq'], images['gt']
        return {'lq': TVF.to_tensor(lq), 'gt': TVF.to_tensor(gt)}

class GoPro(Dataset):
    def __init__(self, root='./dataset', mode='train', new_datasets=False):
        super(GoPro, self).__init__()
        # 1. 이미지들의 경로 저장
        # 2. 이미지 전처리 옵션 설정
        if new_datasets:
            self.lqs = sorted(glob(f'{root}/new_{mode}/blur_crops/*'))
            self.gts = sorted(glob(f'{root}/new_{mode}/sharp_crops/*'))
        else:
            self.lqs = sorted(glob(f'{root}/{mode}/*/blur/*'))
            self.gts = sorted(glob(f'{root}/{mode}/*/sharp/*'))
            
        self.root = root
        self.mode = mode

        self.transform_train = transforms.Compose([
            # CustomToTensor(),
            CustomRandomHorizontalFlip(p=0.5), # 다양한 이미지를 추출하기 위해 적용
            CustomRandomRotation(degrees=90),    # 256 정방형으로 크기를 고정하고 회전시키는게 좋을거같다는 판단ㄴ
            CustomRandomCrop((256, 256)), # @@@ Resize는 사진을 변형시켜서, 나중에 256x256으로 resize한 사진에만 훈련효과를 볼수있음
        ])

        self.transform_test = transforms.Compose([
            # transforms.CenterCrop(224),
            CustomRandomCrop((256, 256)), # @@@ Resize는 사진을 변형시켜서, 나중에 256x256으로 resize한 사진에만 훈련효과를 볼수있음
        ])


    def __getitem__(self, index):
        lq = transforms.ToTensor()(Image.open(self.lqs[index]).convert('RGB'))
        gt = transforms.ToTensor()(Image.open(self.gts[index]).convert('RGB'))
        
        if self.mode == 'train': images = self.transform_train({'lq':lq, 'gt':gt})
        elif self.mode == 'test': images = self.transform_test({'lq':lq, 'gt':gt})
            
        return images['lq'], images['gt']

    def __len__(self):
        return len(self.lqs)

class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        
    def forward(self, pred, target):
        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()

    def forward(self, input, target):
        diff = torch.fft.fft2(input) - torch.fft.fft2(target)
        loss = torch.mean(abs(diff))
        return loss

def main():
    # Dataloader
    batch_size = 6
    shuffle = True

    # Network
    img_channel = 3
    width = 32
    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    # Loss
    losses = ['L1Loss', 'PSNRLoss', 'FFTLoss']
    loss = 'FFTLoss'
    loss_folders = [f'checkpoint/{loss}', f'img/result/{loss}']
    for loss_folder in loss_folders:
        if not os.path.exists(loss_folder): os.makedirs(loss_folder)

    # Train
    resume_train = True
    try: curr_epoch = list(map(lambda x : x[-8:-4], glob(f'checkpoint/{loss}/*')))[-1]
    except: curr_epoch = '0000'; resume_train = False
    epochs = 1000

    # Dataloading
    train_ds = GoPro(root='./dataset', mode='train', new_datasets=True)
    test_ds = GoPro(root='./dataset', mode='test')
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, drop_last=True)

    # Network
    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                        enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    net.to(device)

    # Loss
    if loss == 'L1Loss':
        cri_pix = nn.L1Loss()
    elif loss == 'PSNRLoss':
        cri_pix = PSNRLoss()
    elif loss == 'FFTLoss':
        # cri_pix = PSNRLoss()
        cri_pix = nn.L1Loss()
        cri_pix_fft = FFTLoss()
        cri_pix_fft.to(device)
    else:
        cri_pix = nn.L1Loss()

    cri_pix.to(device)

    # Optimizer
    optimizer_g = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3, betas=(0.9, 0.9))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=int(len(train_ds)/batch_size), eta_min=1e-7)


    try: curr_epoch = list(map(lambda x : x[-8:-4], glob(f'checkpoint/{loss}/*')))[-1]
    except: curr_epoch = '0000'; resume_train = False

    if resume_train:
        state_dict = torch.load(f'./checkpoint/{loss}/checkpoint.{curr_epoch}.pth')

        resume_epoch = state_dict['epoch'] + 1  # +1 부터 시작할 수 있게
        optimizer_state_dict = state_dict['optimizer_state_dict']
        model_state_dict = state_dict['model_state_dict']
        scheduler_state_dict = state_dict['scheduler_state_dict']

        print(f'resume_epoch: {resume_epoch}')

        optimizer_g.load_state_dict(optimizer_state_dict)
        net.load_state_dict(model_state_dict, strict=True)
        scheduler.load_state_dict(scheduler_state_dict)
    else:
        resume_epoch = 0

    for epoch in range(resume_epoch, epochs):
        batch_cnt = 0
        pbar = tqdm(train_dl)
        for i, batch in enumerate(pbar):
            batch_cnt += 1

            lq, gt = batch
            lq = lq.to(device)
            gt = gt.to(device)
            
            optimizer_g.zero_grad()

            preds_t = net(lq)
            if not isinstance(preds_t, list): # preds가 tensor라서 이걸 list로 바꿔주는거
                preds = [preds_t]

            output = preds[-1]

            # pixel loss
            l_total = 0.
            l_pix = 0.
            l_pix_fft = 0.
            for pred in preds:
                l_pix += cri_pix(pred, gt)   # 누적연산은 좋지않댔는데, float를 붙이면 해결가능함 (https://pytorch.org/docs/stable/notes/faq.html#my-model-reports-cuda-runtime-error-2-out-of-memory)
                #----------------------------------------------------------#
                l_pix_fft += cri_pix_fft(pred, gt)
                #----------------------------------------------------------#
            
            l_total += ((0.05 * l_pix_fft) + l_pix)

            l_total = l_total + 0. * sum(p.sum() for p in net.parameters())
            pbar.set_postfix_str(f"epoch: {epoch}, l_pix: {l_pix}, l_pix_fft: {0.05 * l_pix_fft}, l_total: {l_total} lr: {optimizer_g.param_groups[0]['lr']}")

            l_total.backward()
            use_grad_clip = True
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01)
            optimizer_g.step()

            if batch_cnt % 300 == 0:
                img_sample = torch.cat((lq.data, preds_t, gt.data), -2) # 높이(height)를 기준으로 이미지를 연결하기
                save_image(img_sample, f"img/result/{loss}/result{epoch:04d}_{batch_cnt}.png", nrow=batch_size, normalize=False)    
        
        scheduler.step()

        img_sample = torch.cat((lq.data, preds_t, gt.data), -2) # 높이(height)를 기준으로 이미지를 연결하기
        save_image(img_sample, f"img/result/{loss}/result{epoch:04d}.png", nrow=batch_size, normalize=False)
        print(f"[Sample img 저장 완료] img/result/{loss}/result{epoch:04d}.png")
        
        if epoch % 5 == 0:
            torch.save({
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer_g.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict(),
                    'model_state_dict': net.state_dict(),
                }, f'./checkpoint/{loss}/checkpoint.{epoch:04d}.pth')
            print(f'[Check point 저장 완료] checkpoint/{loss}/checkpoint.{epoch:04d}.pth')


if __name__ == "__main__":
    main()