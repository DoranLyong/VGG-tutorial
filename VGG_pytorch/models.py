
#%%
import torch 
import torch.nn as nn  # 학습 가능한 레이어들을 담은 패키지 ; # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F # 학습 안 되는 레이어들을 담은 패키지 ; # All functions that don't have any parameters, relu, tanh, etc. 


VGG_types = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# After then, flatten and '4096x4096xnum_classes' Linear Layers 


#%%
class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__()  # 상속받은 부모 클래스의 모든 attributes을 그대로 받아옴 
        
        """ Conv. layers 정의 
        """
        self.in_channels = in_channels # 첫 입력 데이터의 채널 길이 
        self.conv_layers = self.create_conv_layers(VGG_types['VGG16'])  #정의된 아키텍쳐의 모든 레이어를 한 곳으로 가져오기 
        

        """ Flatten and Linear layers 정의 
        fully-connected layers 
        """
        self.fcs = nn.Sequential(   nn.Linear(512*7*7, 4096),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(4096, 4096),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(4096, num_classes),
                                )

    def forward(self, x):
        x = self.conv_layers(x) # 입력 := [1, 3, 224, 224]  -> 출력 := [1, 512, 7, 7]
        x = x.reshape(x.shape[0], -1)   # for flatten
                                        # x.shape[0] 부분은 Batch_channel
        x = self.fcs(x)
        return x 


    def create_conv_layers(self, architecture):
        """ 좀더 네트워크 생성을 일반화 시키기 위해 (옵션 입력을 가변적으로 받아서 VGG11, VGG13... 을 생성)
        """
        layers = []  # 레이어 아키텍쳐 초기화 
        in_channels = self.in_channels


        for x in architecture:
            if type(x) == int: 
                """ 순회하면서 Conv 레이어를 쌓는 과정. 
                    64, 128, 256, 512 레이버 부분만 쌓음 
                """
                out_channels = x  # 해당 레이어에서 출력하는 feature_map의 채널 길이 

                layers += [ nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                            nn.BatchNorm2d(x),
                            nn.ReLU(),
                            ]

                in_channels = x  # 출력된 feature_map은 다음 레이어에서 입력으로 들어가니까 

            elif x == 'M':
                """ MaxPooling
                feature_map 의 hight, width 사이즈만 줄어들지, 
                채널 길이는 그대로 유지됨 
                """
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

        
        # 전체 정의된 요소들을 하나의 시퀀스로 엮음; 사람이 보기에 가독성이 좋아짐 
        # (ref) https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
        # Unpacking the list 
        return nn.Sequential(*layers)



# %% 
if __name__ == "__main__":

    # 프로세스 장비 설정 
    gpu_no = 0
    device = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')

    # 모델 초기화 
    model = VGG_net(in_channels=3,num_classes=10).to(device)
    print(model)


    # 모델 출력 테스트 
    x = torch.randn(1, 3, 224, 224).to(device)
    print(model(x).shape)

# %%
