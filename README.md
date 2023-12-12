# classification
- 예측할 값이 정해져 있는 경우. ==> 범주형인 경우.
- 다중분류 : 범주값(class)가 여러인 경우
- 이진분류 : 범주값이 0 or 1 => 맞는지 틀린지를 추정 문제. (맞는것: Posivitve -> 1, 틀린것: Negative -> 0)

## Fashion MNIST Dataset - 다중분류(Multi-Class Classification) 문제
- 10개의 범주(category)와 70,000개의 흑백 이미지로 구성된 패션 MNIST 데이터셋. 이미지는 해상도(28x28 픽셀)가 낮고 다음처럼 개별 의류 품목을 나타낸다:

## 1. import 
```python
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torchinfo

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
```

## 2. dataset 생성
```python
fmnist_trainset = datasets.FashionMNIST(root="datasets", train=True, download=True,
                                       transform=transforms.ToTensor())
fmnist_testset = datasets.FashionMNIST(root="datasets", train=False, download=True,
                                      transform=transforms.ToTensor())
```
### 데이터셋 확인
```python
# index to class 
index_to_class = np.array(fmnist_trainset.classes) # fany indexing을 위해서 list->ndarray
index_to_class[[1, 1, 2, 3, 0]]

array(['Trouser', 'Trouser', 'Pullover', 'Dress', 'T-shirt/top'],
      dtype='<U11')

# class to index
class_to_index = fmnist_trainset.class_to_idx
class_to_index

{'T-shirt/top': 0,
 'Trouser': 1,
 'Pullover': 2,
 'Dress': 3,
 'Coat': 4,
 'Sandal': 5,
 'Shirt': 6,
 'Sneaker': 7,
 'Bag': 8,
 'Ankle boot': 9}
```
## 3. dataloader 생성
```python
### DataLoader
fmnist_trainloader = DataLoader(fmnist_trainset, batch_size=128, shuffle=True,
                                drop_last=True)
fmnist_testloader = DataLoader(fmnist_testset, batch_size=128)
```

## 4. 모델정의
```python
class FashionMNISTModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # 입력 이미지를 받아서 처리후 리턴
        self.lr1 = nn.Linear(28*28, 2048) # 784  -> 2048
        self.lr2 = nn.Linear(2048, 1024)  # 2048 -> 1024
        self.lr3 = nn.Linear(1024, 512)   # 1024 ->  512
        self.lr4 = nn.Linear(512, 256)    # 512  ->  256
        self.lr5 = nn.Linear(256, 128)    # 256  ->  128
        self.lr6 = nn.Linear(128, 64)     # 128  ->   64
        # output - out_features: 다중분류- class 개수 (fashion mnist: 10)
        self.lr7 = nn.Linear(64, 10)  # 각 클래스별 확률이 출력되도록 한다.
    def forward(self, X):
        #  X: (batch, channel, height, width) ====> (batch, channel*height*width)
        #  out = torch.flatten(X, start_dim=1)
        out = nn.Flatten()(X)
        # lr1  ~ lr7
        ## forward 처리를 구현. => Linear -> ReLU() (lr7의 출력은 ReLU에 넣지 마세요.)
        out = nn.ReLU()(self.lr1(out))
        out = nn.ReLU()(self.lr2(out))
        out = nn.ReLU()(self.lr3(out))
        out = nn.ReLU()(self.lr4(out))
        out = nn.ReLU()(self.lr5(out))
        out = nn.ReLU()(self.lr6(out))
        ### output
        out = self.lr7(out)
        return out
```
### 모델 생성 및 확인
```python
## 모델 생성 및 확인
f_model = FashionMNISTModel()
print(f_model)

FashionMNISTModel(
  (lr1): Linear(in_features=784, out_features=2048, bias=True)
  (lr2): Linear(in_features=2048, out_features=1024, bias=True)
  (lr3): Linear(in_features=1024, out_features=512, bias=True)
  (lr4): Linear(in_features=512, out_features=256, bias=True)
  (lr5): Linear(in_features=256, out_features=128, bias=True)
  (lr6): Linear(in_features=128, out_features=64, bias=True)
  (lr7): Linear(in_features=64, out_features=10, bias=True)
)
```

## 5. 학습(train)
```python
## 하이퍼파라미터
LR = 0.001
N_EPOCH = 20

# 모델을 device 로 이동.
f_model = f_model.to(device)
# loss fn -> 다중분류: nn.CrossEntropyLoss() ==> 다중 분류용 Log loss 
loss_fn = nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.Adam(f_model.parameters(), lr=LR)
```
```python
# train
import time
## 각 에폭별 학습이 끝나고 모델 평가한 값을 저장.
train_loss_list = []
valid_loss_list = []
valid_acc_list = []   # test set의 정확도 검증 결과 => 전체데이터 중 맞은데이터의 개수

s = time.time()
for epoch in range(N_EPOCH):
    ######### train
    f_model.train()
    train_loss=0.0
    for X_train, y_train in fmnist_trainloader:
        # 1. 디바이스 옮기기
        X_train, y_train = X_train.to(device), y_train.to(device) # 모델과 같은 디바이스로 옮겨야함
        # 2. 예측
        pred_train = f_model(X_train)
        # 3. 오차 계산
        loss = loss_fn(pred_train, y_train)
        # 4. 모델 파라미터 업데이터
        ## 4-1 gradient 초기화
        optimizer.zero_grad()
        ## 4-2 grad 계산
        loss.backward()
        ## 4-3 파라미터 업데이트
        optimizer.step()
        train_loss += loss.item() # train_loss 누적 / item() : 텐서의 들어가있는 값을 파이썬값으로 빼온다 
    # 1에폭 종료 -> train_loss의 평균을 list에 저장
    train_loss /= len(fmnist_trainloader) # 누적 train_loss에 step수만큼 나눈다
    train_loss_list.append(train_loss)
    
    
    ######### validation
    f_model.eval()
    valid_loss = 0.0 # 현재 epoch의 validation loss 저장할 변수
    valid_acc = 0.0 # 현재 epoch의 validation accuracy 저장할 변수
    with torch.no_grad(): # 평가만 하기 때문에 파라미터 업데이트하고 그럴 필요 없다
        for X_valid, y_valid in fmnist_testloader:
            X_valid, y_valid = X_valid.to(device), y_valid.to(device)
            
            pred_valid = f_model(X_valid) # 클래스별 정답일 가능성 (batch, 10)
            pred_label = pred_valid.argmax(dim=-1) # 정답 클래스 조회 (pred_valid에서 가장 큰 값을 가진 인덱스)
            
            valid_loss += loss_fn(pred_valid, y_valid).item()
            valid_acc += torch.sum(pred_label == y_valid).item()
        # 한 epoch에 대한 평가 완료 -> list에 추가
        valid_loss /= len(fmnist_testloader)  # step수로 나눠서 평균 계산
        valid_acc /= len(fmnist_testloader.dataset) # testset의 총 데이터 개수로 나눔
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)
    
        print(f"[{epoch+1}/{N_EPOCH}] train loss: {train_loss:.4f}, valid loss: {valid_loss:.4f}, valid acc: {valid_acc:.4f}")
    
    
    
e = time.time()
print('걸린시간:', (e-s), "초")
```
## 6. 결과 시각화
```python
# 결과 시각화
plt.rcParams['font.family'] = 'Malgun gothic'
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label='train')
plt.plot(valid_loss_list, label='validation')
plt.title('epoch별 loss 변화')
plt.legend()

plt.subplot(1,2,2)
plt.plot(valid_acc_list)
plt.title('validation accuracy')

plt.tight_layout()
plt.show()
```
![12](https://github.com/jsj5605/classification/assets/141815934/0ec13e82-67c6-4578-824f-ec1d90ca8c05)

## 7. 성능 개선(조기종료)
```python
# 모델 학습이 진행되면 어느 시점부터 성능이 떨어지기 시작 
# trainset으로 검증한 결과는 계속 좋아지는데 validation set으로 검증한 결과는 성능이 좋아지다가 안좋아진다
# 1. 학습도중 성능이 개선될 때마다 저장(가장 좋은 성능의 모델을 서비스 할 수 있다)
# 2. 더이상 성능개선이 안되면 학습을 중지(조기종료)

##############################################
# 조기종료 + 모델 저장을 위한 변수 추가
#############################################
## 모델 저장을 위한 변수
# 학습 중 가장 좋은 성능 저장. 현 epoch의 지표가 이 변수값보다 좋으면 저장
# 평가지표 : validation loss
best_score = torch.inf
save_model_path = 'models/fashion_mnist_best_model.pth'

## 조기 종료를 위한 변수: 특정 epoch동안 성능개선이 없으면 학습을 중단
patience = 5 # 성능이 개선될지를 기다릴 epoch 수. patience만큼 개선이 안되면 중단 (보통 10이상 지정)
trigger_cnt = 0 # 성능 개선을 몇번째 기다리는지 정할 변수 patience == trigger_cnt 일때 중단

##############################
# 조기종료여부, 모델 저장 처리
# 저장 : 현 epoch valid_loss가 best_score보다 개선된 경우 저장(작으면 개선)
##############################
if valid_loss < best_score: # 성능이 개선
# 저장 로그 출력
  print(f"모델저장: {epoch+1}epoch - 이전 valid loss: {best_score}, 현재 valid loss: {valid_loss}")
  # best_score 교체
  best_score = valid_loss
  # 저장
  torch.save(f_model, save_model_path)
  # trigger_cnt를 0으로 초기화
  trigger_cnt = 0
else: # 성능개선이 안된경우
  # trigger_cnt 1증가
  trigger_cnt += 1
  if patience == trigger_cnt:
    # 로그
    print(f"-> {epoch+1} epoch에서 조기종료 - {best_score}에서 개선안됨")
    break
```
## 8. 저장된 모델로 테스트
```python
# 저장된 모델 로딩
best_model = torch.load(save_model_path)

# test_dataloader로 평가
best_model = best_model.to(device)
best_model.eval()
valid_loss = 0.0 # 현재 epoch의 validation loss 저장할 변수
valid_acc = 0.0 # 현재 epoch의 validation accuracy 저장할 변수
with torch.no_grad(): # 평가만 하기 때문에 파라미터 업데이트하고 그럴 필요 없다
    for X_valid, y_valid in fmnist_testloader:
        X_valid, y_valid = X_valid.to(device), y_valid.to(device)
        
        pred_valid = best_model(X_valid) # 클래스별 정답일 가능성 (batch, 10)
        pred_label = pred_valid.argmax(dim=-1) # 정답 클래스 조회 (pred_valid에서 가장 큰 값을 가진 인덱스)
        
        valid_loss += loss_fn(pred_valid, y_valid).item()
        valid_acc += torch.sum(pred_label == y_valid).item()
    # 한 epoch에 대한 평가 완료 -> list에 추가
    valid_loss /= len(fmnist_testloader)  # step수로 나눠서 평균 계산
    valid_acc /= len(fmnist_testloader.dataset) # testset의 총 데이터 개수로 나눔

valid_loss : 0.3155972200292575, valid_acc : 0.8915
```











           
