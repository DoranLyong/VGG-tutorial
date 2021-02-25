#-*- coding: utf-8 -*-

"""
(ref) https://youtu.be/ACmuBbuXn20
(ref) https://github.com/aladdinpersson/Machine-Learning-Collection/blob/79f2e1928906f3cccbae6c024f3f79fd05262cd1/ML/Pytorch/CNN_architectures/pytorch_vgg_implementation.py
(ref) 포멧 따라하기; https://github.com/DoranLyong/ResNet-tutorial/blob/main/ResNet_pytorch/ResNet18_for_CIFAR-10.py
(ref) train-loop 따라하기; https://github.com/DoranLyong/DeepLearning_model_factory/blob/master/ML_tutorial/PyTorch/Basics/lr_scheduler_tutorial.py

* Run this code on VScode (it can be run with Jupyter notebook conntection)
* Run each code cell with pressing 'shift + Enter'   
* Like here -> https://blog.naver.com/cheeryun/221685740941
"""

"""
VGG16 네트워크를 활용해 CIFAR10 분류하기 
    * VGG16 : 학습 가능한 weight가 있는 레이어가 16개 있다는 의미 

1. Create VGG16 model  

2. Set device 

3. Hyperparameters 

4. Load Data (CIFAR10)

5. Initialize network 

6. Loss and optimizer 

7. Train network 

8. Check accuracy on training & test to see how good our model
"""

#%% 임포트 토치 
import os.path as osp
import os

from tqdm import tqdm 
import torch 
import torch.nn as nn  # 학습 가능한 레이어들을 담은 패키지 ; # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.backends.cudnn as cudnn    # https://hoya012.github.io/blog/reproducible_pytorch/
                                        # https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
import torch.optim as optim  # 최적화 알고리즘을 담은 패키지 ; # For all Optimization algorithms, SGD, Adam, etc.
from torch.utils.data import DataLoader   # Gives easier dataset management and creates mini batches

import torchvision.datasets as datasets  # 이미지 데이터를 불러오고 변환하는 패키지 ;  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset

from models import VGG_net


# ================================================================= #
#                         2. Set device                             #
# ================================================================= #
# %% 02. 프로세스 장비 설정 
gpu_no = 0  # gpu_number 
device = torch.device( f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")



# ================================================================= #
#                       3. Hyperparameters                          #
# ================================================================= #
# %% 03. 하이퍼파라미터 설정 
num_classes = 10 
learning_rate = 1e-4
batch_size = 32   # 메모리가 부족하면 사이즈를 줄일 것 
NUM_EPOCHS = 40

SEED = 42 # set seed 
load_model = True  # 체크포인트 모델을 가져오려면 True 



# ================================================================= #
#                      4.  Load Data (CIFAR10)                        #
# ================================================================= #
# %% 04. CIFAR10 데이터 로드 
"""
랜덤 발생 기준을 시드로 고정함. 
그러면 shuffle=True 이어도, 언제나 동일한 방식으로 섞여서 동일한 데이터셋을 얻을 수 있음. 
"""
torch.manual_seed(42)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) # for multi-gpu


transform_train = transforms.Compose([ #Compose makes it possible to have many transforms
                                        transforms.Resize((224,224)), # Resizes (32,32) to (224, 224) for input to VGG16 network 
                                        transforms.ToTensor(), # 데이터 타입을 Tensor 형태로 변경 ; (ref) https://mjdeeplearning.tistory.com/81 # Finally converts PIL image to tensor so we can train w. pytorch
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Note: these values aren't optimal,
                                    ])

transform_test = transforms.Compose([   transforms.Resize((224,224)), # Resizes (32,32) to (224, 224) for input to VGG16 network 
                                        transforms.ToTensor(),
                                    ])



train_dataset = datasets.CIFAR10( root='dataset/',    # 데이터가 위치할 경로 
                                train=True,         # train 용으로 가져오고 
                                transform=transform_train,  
                                download=True       # 해당 root에 데이터가 없으면 torchvision 으로 다운 받아라 
                                )

train_loader = DataLoader(  dataset=train_dataset,   # 로드 할 데이터 객체 
                            batch_size=batch_size,   # mini batch 덩어리 크기 설정 
                            shuffle=True,            # 데이터 순서를 뒤섞어라 
                            num_workers=4,           # Broken Pipe 에러 뜨면 지우기 
                            )      


test_dataset = datasets.CIFAR10(root='dataset/', train=False, transform=transform_test, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)



# ================================================================= #
#                    5.  Initialize the network                     #
# ================================================================= #
# %% 05. 모델 초기화 
model = VGG_net(in_channels=3,num_classes=10).to(device)
print(model)

model = torch.nn.DataParallel(model)# 데이터 병렬처리          # (ref) https://tutorials.pytorch.kr/beginner/blitz/data_parallel_tutorial.html
                                    # 속도가 더 빨라지진 않음   # (ref) https://tutorials.pytorch.kr/beginner/former_torchies/parallelism_tutorial.html
                                    # 오히려 느려질 수 있음    # (ref) https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b
cudnn.benchmark = True

# %%
# ================================================================= #
#                       6.  Loss and optimizer                      #
# ================================================================= #
# %% 06. 손실 함수와 최적화 알고리즘 정의 
criterion = nn.CrossEntropyLoss()   
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 네트워크의 모든 파라미터를 전달한다 


# ================================================================= #
#                       6-1.  Define Scheduler                      #
# ================================================================= #
# %% 스케쥴러 정의 

# Define Scheduler
"""
(ref) https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
"""
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1, patience=5, verbose=True)






# ================================================================= #
#                              Train-loop                           #
# ================================================================= #

# %% Train-loo  정의 
def train(epoch):

    model.train()  

    train_losses = [] 
    num_correct = 0 
    num_samples = 0 

    # (ref) https://github.com/DoranLyong/DeepLearning_model_factory/blob/master/ML_tutorial/PyTorch/Basics/Utility/progress_bar_for_DataLoader.py
    loop = tqdm(enumerate(train_loader), total=len(train_loader))  
    

    for batch_idx, (data, targets) in loop: # 미니배치 별로 iteration 
        """Get data to cuda if possible
        """
        data = data.to(device=device)  # 미니 베치 데이터를 device 에 로드 
        targets = targets.to(device=device)  # 레이블 for supervised learning 

        """forward
        """
        scores = model(data)   # 모델이 예측한 수치 
        loss = criterion(scores, targets)
        train_losses.append(loss.item())

        _, predictions = scores.max(1)
        num_correct += predictions.eq(targets).sum().item()   #  맞춘 샘플 개수 
        num_samples += predictions.size(0)   # 총 예측한 샘플 개수 (맞춘 것 + 틀린 것)

        """backward
        """
        optimizer.zero_grad()   # AutoGrad 하기 전에(=역전파 실행전에) 매번 mini batch 별로 기울기 수치를 0으로 초기화 
        loss.backward()         # (ref) https://tutorials.pytorch.kr/beginner/pytorch_with_examples.html
        
        """gradient descent or adam step
        """
        optimizer.step()
        

        """ progress bar with tqdm
        """
        # (ref) https://github.com/DoranLyong/DeepLearning_model_factory/blob/master/ML_tutorial/PyTorch/Basics/Utility/progress_bar_for_DataLoader.py
        # (ref) https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/lr_scheduling/
        # LR 수치 가져와 보기 ; (ref) https://discuss.pytorch.org/t/how-to-retrieve-learning-rate-from-reducelronplateau-scheduler/54234

        loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}], LR={ optimizer.param_groups[0]['lr'] :.1e}")
        

        if batch_idx % batch_size == 0:  
            loop.set_postfix( acc=(predictions == targets).sum().item() / predictions.size(0), loss=loss.item(),  batch=batch_idx)


    total_acc = float(num_correct)/float(num_samples)*100
    mean_loss = sum(train_losses) / len(train_losses)

    print(f"\nTotal train acc: {total_acc:.2f}%")
    print(f"Mean loss of train: {mean_loss:.5f}") # 소수 다섯째 자리까지 표시 



    """After each epoch do scheduler.step, 
        note in this scheduler we need to send in loss for that epoch!
        lr 수치가 바뀌면 뭐가 뜨나? 
    """
    scheduler.step(mean_loss) # Decay Learning Rate
    


# %% Train-loop 실행 
for epoch in range(1, NUM_EPOCHS+1):
    train(epoch)
