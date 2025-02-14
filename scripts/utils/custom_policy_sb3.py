
from tracemalloc import start
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch as th
from torch.nn.modules.linear import Linear

import torchvision.models as pre_models
import numpy as np
import torch.nn.functional as F

'''
Here we provide 5 feature extractor networks

1. No_CNN
    No CNN layers
    Only maxpooling layer to generate 25 features

2. CNN_GAP
    3 layers of CNN
    finished by AvgPool2d
    1*8 -> 8*16 -> 16*25

3. CNN_GAP_BN
    3 layers of CNN with BN for each CNN layer
    finished by AvgPool2d

4. CNN_FC
    3 layers of CNN
    finished by Flatten
    FC is used to get CNN features (960 100 25)

5. CNN_MobileNet
    Using a pre-trained MobileNet as feature generator
    finished by Flatten (576 -> 25)

6. TransformerFeatureExtractor

'''


class No_CNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=4):
        super(No_CNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # Can use model.actor.features_extractor.feature_all to print all features

        # set CNN and state feature num
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_all = None

        # input size 80*100
        # divided by 5

        self.cnn = nn.Sequential(
            nn.MaxPool2d(kernel_size=(12, 18)),

            #nn.MaxPool2d(kernel_size=(16, 20)),
            #nn.MaxPool2d(kernel_size=(26, 33)),
            nn.Flatten()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]
        #print(depth_img.shape)
        cnn_feature = self.cnn(depth_img)  # [1, 25, 1, 1]
        #print(cnn_feature)
        #print(self.feature_num_state)

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1
        # print(state_feature.size(), cnn_feature.size())
        x = th.cat((cnn_feature, state_feature), dim=1)
        #print("custom_policy_NO_CNN:x.shape:",x.shape)
        

        self.feature_all = x  # use  to update feature before F

        return x


class CNN_GAP(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=0):
        super(CNN_GAP, self).__init__(observation_space, features_dim)
        # Can use model.actor.features_extractor.feature_all to print all features
        # set CNN and state feature num
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim
        self.feature_all = None

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # [1, 8, 40, 50]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 16, 20, 25]
            # nn.BatchNorm2d(8, affine=False)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, self.feature_num_cnn, kernel_size=3,
                      stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, self.feature_num_cnn, 10, 12]
        )
        self.gap_layer = nn.AvgPool2d(kernel_size=(10, 12), stride=1)

        self.batch_layer = nn.BatchNorm1d(self.feature_num_cnn)

        # nn.init.kaiming_normal_(self.conv1[0].weight, a=0, mode='fan_in')
        # nn.init.kaiming_normal_(self.conv2[0].weight, a=0, mode='fan_in')
        # nn.init.kaiming_normal_(self.conv3[0].weight, a=0, mode='fan_in')
        # nn.init.constant(self.conv1[0].bias, 0.0)
        # nn.init.constant(self.conv2[0].bias, 0.0)
        # nn.init.constant(self.conv3[0].bias, 0.0)

        # nn.init.xavier_uniform(self.conv1[0].weight)
        # nn.init.xavier_uniform(self.conv2[0].weight)
        # nn.init.xavier_uniform(self.conv3[0].weight)
        # self.conv1[0].bias.data.fill_(0)
        # self.conv2[0].bias.data.fill_(0)
        # self.conv3[0].bias.data.fill_(0)
        # self.soft_max_layer = nn.Softmax(dim=1)
        # self.batch_norm_layer = nn.BatchNorm1d(16, affine=False)

        # self.linear = self.cnn

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]  # [1, 1, 80, 100]
        print("depth_img.shape",depth_img.shape)
        self.layer_1_out = self.conv1(depth_img)
        self.layer_2_out = self.conv2(self.layer_1_out)
        self.layer_3_out = self.conv3(self.layer_2_out)
        self.gap_layer_out = self.gap_layer(self.layer_3_out)

        cnn_feature = self.gap_layer_out  # [1, 8, 1, 1]
        print("cnn_feature.shape:",cnn_feature.shape)

        cnn_feature = cnn_feature.squeeze(dim=3)  # [1, 8, 1]
        cnn_feature = cnn_feature.squeeze(dim=2)  # [1, 8]
        # cnn_feature = th.clamp(cnn_feature,-1,2)
        # cnn_feature = self.batch_layer(cnn_feature)

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1

        x = th.cat((cnn_feature, state_feature), dim=1)
        print("custom_policy_CNN_GAP:x.shape:",x.shape)


        self.feature_all = x  # use  to update feature before FC

        return x


class CNN_GAP_BN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=0):
        super(CNN_GAP_BN, self).__init__(observation_space, features_dim)
        # Can use model.actor.features_extractor.feature_all to print all features
        # set CNN and state feature num
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim
        self.feature_all = None

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # [1, 8, 40, 50]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 16, 20, 25]
            # nn.BatchNorm2d(8, affine=False)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, self.feature_num_cnn, kernel_size=3,
                      stride=1, padding='same'),
            nn.BatchNorm2d(self.feature_num_cnn),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, self.feature_num_cnn, 10, 12]
        )
        self.gap_layer = nn.AvgPool2d(kernel_size=(10, 12), stride=1)

        self.batch_layer = nn.BatchNorm1d(self.feature_num_cnn)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]

        self.layer_1_out = self.conv1(depth_img)
        self.layer_2_out = self.conv2(self.layer_1_out)
        self.layer_3_out = self.conv3(self.layer_2_out)
        self.gap_layer_out = self.gap_layer(self.layer_3_out)

        cnn_feature = self.gap_layer_out  # [1, 8, 1, 1]
        cnn_feature = cnn_feature.squeeze(dim=3)  # [1, 8, 1]
        cnn_feature = cnn_feature.squeeze(dim=2)  # [1, 8]
        print("cnn_feature.shape:",cnn_feature.shape)
        # cnn_feature = th.clamp(cnn_feature,-1,2)
        # cnn_feature = self.batch_layer(cnn_feature)

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1

        x = th.cat((cnn_feature, state_feature), dim=1)
        self.feature_all = x  # use  to update feature before FC
        print("custom_policy_CNN_GAP_BN:x.shape:",x.shape)

        return x


class CustomNoCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=4):
        super(CustomNoCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # Can use model.actor.features_extractor.feature_all to print all features

        # set CNN and state feature num
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_all = None

        # input size 80*100
        # divided by 5
        self.cnn = nn.Sequential(
            nn.MaxPool2d(kernel_size=(16, 20)),
            nn.Flatten()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]

        cnn_feature = self.cnn(depth_img)  # [1, 25, 1, 1]

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1
        # print(state_feature.size(), cnn_feature.size())
        x = th.cat((cnn_feature, state_feature), dim=1)
        # print(x)
        self.feature_all = x  # use  to update feature before FC
        print("custom_policy_CustomNoCNN:x.shape:",x.shape)


        return x


class CNN_FC(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=0):
        super(CNN_FC, self).__init__(observation_space, features_dim)
        # Can use model.actor.features_extractor.feature_all to print all features
        # set CNN and state feature num
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim
        self.feature_all = None

        # Input image: 80*100
        # Output: 16 CNN features + n state features
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 8, 40, 48]

            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 8, 20, 24]

            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 8, 10, 12]

            # nn.BatchNorm2d(8),
            nn.Flatten(),   # 960
            # nn.AvgPool2d(kernel_size=(10, 12), stride=1)
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[
                             None][:, 0:1, :, :]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 100),
            nn.ReLU(),
            nn.Linear(100, self.feature_num_cnn),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]

        cnn_feature = self.linear(self.cnn(depth_img))
        # cnn_feature = cnn_feature.squeeze(dim=3) # [1, 8, 1]
        # cnn_feature = cnn_feature.squeeze(dim=2) # [1, 8]

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1

        x = th.cat((cnn_feature, state_feature), dim=1)
        self.feature_all = x  # use  to update feature before FC
        print("custom_policy_CNN_FC:x.shape:",x.shape)

        return x


class CNN_MobileNet(BaseFeaturesExtractor):
    '''
    Using part of mobile_net_v3_small to generate features from depth image
    '''

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=0):
        super(CNN_MobileNet, self).__init__(observation_space, features_dim)

        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim
        self.feature_all = None

        self.mobilenet_v3_small = pre_models.mobilenet_v3_small(
            pretrained=True)

        self.part = self.mobilenet_v3_small.features

        # freeze part parameters
        for param in self.part.parameters():
            param.requires_grad = False

        self.gap_layer = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Sequential(
            nn.Linear(576, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(256, self.feature_num_cnn),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Dropout(0.25)
        )
        self.linear_small = nn.Sequential(
            nn.Linear(576, self.feature_num_cnn),
            nn.Tanh(),
            # nn.BatchNorm1d(32),
            # nn.Dropout(0.25)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]

        # change input image to (None, 3, 100, 80)
        # notion: this repeat is used for tensor  # (1, 3, 80 ,100)
        depth_img_stack = depth_img.repeat(1, 3, 1, 1)

        self.last_cnn_output = self.part(
            depth_img_stack)        # [1, 576, 3, 4]
        self.gap_layer_out = cnn_feature = self.gap_layer(
            self.last_cnn_output)  # [1, 576, 1, 1]

        cnn_feature = cnn_feature.squeeze(dim=3)  # [1, 576, 1]
        cnn_feature = cnn_feature.squeeze(dim=2)  # [1, 576]
        cnn_feature = self.linear_small(cnn_feature)  # [1, 32]

        state_feature = observations[:, 1, 0,
                                     0:self.feature_num_state]  # [1, 2]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1

        x = th.cat((cnn_feature, state_feature), dim=1)
        self.feature_all = x  # use  to update feature before FC
        # print(x)
        print("custom_policy_CNN_MobileNet:x.shape:",x.shape)

        return x


class CNN_GAP_new(BaseFeaturesExtractor):

    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim=0):
        super(CNN_GAP_new, self).__init__(observation_space, features_dim)
        # Can use model.actor.features_extractor.feature_all to print all features
        # set CNN and state feature num
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim
        self.feature_all = None

        # input size (100, 80)
        # input size 80 60
        self.conv1 = nn.Conv2d(1, 8, 5, 2)  # 28,38
        self.conv2 = nn.Conv2d(8, 8, 5, 2)  # 12,17
        self.conv3 = nn.Conv2d(8, 8, 3, 2)  # 5, 8
        self.pool = nn.MaxPool2d(2, 3)
        self.gap_layer = nn.AvgPool2d(kernel_size=(8, 10), stride=1)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:1, :, :]  # 0-1 0->20m 1->0m
        # print(th.min(depth_img), th.max(depth_img))
        # norm image to (-1, 1)
        depth_img_norm = (depth_img - 0.5) * 2
        # print(th.min(depth_img_norm), th.max(depth_img_norm))

        # 1, 8, 38, 48  1,8,28,38
        self.layer_1_out = F.relu(self.conv1(depth_img_norm))
        # 1, 8, 18, 23  1,8,12,17
        self.layer_2_out = F.relu(self.conv2(self.layer_1_out))
        self.layer_3_out = F.relu(self.conv3(
            self.layer_2_out))  # 1, 16, 8, 10  1,8,5,8
        self.layer_small = self.pool(self.layer_3_out)  # 1,8,2,3
        # self.gap_layer_out = self.gap_layer(self.layer_3_out)               # 1, 16, 1, 1
        self.flatten = th.flatten(self.layer_small, start_dim=1)
        # self.flatten = self.flatten.unsqueeze(0)

        # cnn_feature = self.gap_layer_out  # [1, 16, 1, 1]
        # cnn_feature = cnn_feature.squeeze(dim=3)  # [1, 16, 1]
        # cnn_feature = cnn_feature.squeeze(dim=2)  # [1, 16]
        # cnn_feature = th.clamp(cnn_feature, -1, 2)
        # cnn_feature = self.batch_layer(cnn_feature)

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1

        x = th.cat((self.flatten, state_feature), dim=1)
        self.feature_all = x  # use  to update feature before FC

        return x


class CNNFeatureExtractor(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # [1, 8, 40, 50]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 16, 20, 25]
            # nn.BatchNorm2d(8, affine=False)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 128, kernel_size=3,
                      stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, self.feature_num_cnn, 10, 12]
        )
        self.gap_layer = nn.AvgPool2d(kernel_size=(10, 12), stride=1)

        self.batch_layer = nn.BatchNorm1d(128)

        # nn.init.kaiming_normal_(self.conv1[0].weight, a=0, mode='fan_in')
        # nn.init.kaiming_normal_(self.conv2[0].weight, a=0, mode='fan_in')
        # nn.init.kaiming_normal_(self.conv3[0].weight, a=0, mode='fan_in')
        # nn.init.constant(self.conv1[0].bias, 0.0)
        # nn.init.constant(self.conv2[0].bias, 0.0)
        # nn.init.constant(self.conv3[0].bias, 0.0)

        # nn.init.xavier_uniform(self.conv1[0].weight)
        # nn.init.xavier_uniform(self.conv2[0].weight)
        # nn.init.xavier_uniform(self.conv3[0].weight)
        # self.conv1[0].bias.data.fill_(0)
        # self.conv2[0].bias.data.fill_(0)
        # self.conv3[0].bias.data.fill_(0)
        # self.soft_max_layer = nn.Softmax(dim=1)
        # self.batch_norm_layer = nn.BatchNorm1d(16, affine=False)

        # self.linear = self.cnn

    def forward(self, observations: th.Tensor) -> th.Tensor:
        depth_img = observations[:, 0:3, :, :]
        print(depth_img.shape)

        self.layer_1_out = self.conv1(depth_img)
        self.layer_2_out = self.conv2(self.layer_1_out)
        self.layer_3_out = self.conv3(self.layer_2_out)
        self.gap_layer_out = self.gap_layer(self.layer_3_out)

        cnn_feature = self.gap_layer_out  # [1, 8, 1, 1]
        print("cnn_feature.shape:",cnn_feature.shape)

        cnn_feature = cnn_feature.squeeze(dim=3)  # [1, 8, 1]
        x = cnn_feature.squeeze(dim=2)  # [1, 8]
        print("custom_policy_CNN_GAP:x.shape:",x.shape)


        self.feature_all = x  # use  to update feature before FC

        return x


class Transformer(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim=256, transformer_dim=128, state_feature_dim = 0, num_layers=2, num_heads=4):
        super(Transformer, self).__init__(observation_space, features_dim)
        
        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim
        self.feature_all = None

        self.cnn = CNNFeatureExtractor()

        # Transformer 层
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(transformer_dim, self.feature_num_cnn)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]

        # CNN 提取特征
        cnn_out = self.cnn(observations)  # shape: (batch_size, transformer_dim)

        # Transformer 需要 (seq_len, batch_size, feature_dim) 格式
        cnn_out = cnn_out.unsqueeze(0)  # 增加时间序列维度: (1, batch_size, transformer_dim)

        # Transformer 处理
        transformer_out = self.transformer(cnn_out)
        transformer_out = self.fc(transformer_out.squeeze(0))
        print("transformer:transformer_out.shape", transformer_out.shape)

        state_feature = observations[:, 1, 0, 0, 0:self.feature_num_state]

        x = th.cat((transformer_out, state_feature), dim = 1)
        self.feature_all = x
        print("transformer:x.shape", x.shape)
        return x

class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    自定义的 Transformer 特征提取器，将观测数据经过全连接层映射到隐藏维度后，
    通过 Transformer Encoder 进行特征提取，最后输出指定维度的特征。
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, state_feature_dim = 0,
                 hidden_dim: int = 128, nhead: int = 4, num_layers: int = 2):
        # 调用父类构造函数，features_dim 表示最终输出的特征维度
        super(TransformerFeaturesExtractor, self).__init__(observation_space, features_dim)

        assert state_feature_dim > 0
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim
        self.feature_all = None

        self.channels = observation_space.shape[0]  # e.g., 3
        self.height = observation_space.shape[1]      # 60
        self.width = observation_space.shape[2]  

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # [1, 8, 40, 50]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 16, 20, 25]
            # nn.BatchNorm2d(8, affine=False)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, hidden_dim, kernel_size=3,
                      stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, self.feature_num_cnn, 10, 12]
        )
        self.gap_layer = nn.AvgPool2d(kernel_size=(10, 12), stride=1)

        self.batch_layer = nn.BatchNorm1d(hidden_dim)

        # 构造 Transformer Encoder：输入维度为 hidden_dim，序列长度为 num_patches
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 最后将 Transformer 输出聚合并映射到 features_dim
        self.fc = nn.Linear(hidden_dim, self.feature_num_cnn)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        observations 的形状为 [batch_size, observation_dim]
        Transformer 需要的输入形状为 [sequence_length, batch_size, d_model]，
        此处我们可以将每个观测视为长度为1的序列。
        """
        depth_img = observations[:, 0:1, :, :]  # [1, 1, 80, 100]
        
        #print("depth_img.shape",depth_img.shape) 

        self.layer_1_out = self.conv1(depth_img)
        #print("1")
        self.layer_2_out = self.conv2(self.layer_1_out)
        #print("2")
        self.layer_3_out = self.conv3(self.layer_2_out)
        #print("3")
        self.gap_layer_out = self.gap_layer(self.layer_3_out)
        #print("4, self.gap_layer_out.shape", self.gap_layer_out.shape)
        #self.batch_layer_out = self.batch_layer(self.gap_layer_out)
        #print("self.batch_layer_out.shape", self.batch_layer_out.shape)
        batch_size, hidden_dim, h, w = self.gap_layer_out.shape
        transformer_feature = self.gap_layer_out.view(batch_size, hidden_dim, -1)   # [batch, hidden_dim, num_patches]
        #print("5, transformer_feature.shape", transformer_feature.shape)
        
        transformer_feature = transformer_feature.permute(2, 0, 1) 
        #print("6, transformer_feature.shape", transformer_feature.shape)


        transformer_feature = self.transformer_encoder(transformer_feature)
        #print("7, transformer_feature.shape", transformer_feature.shape)

                  # [1, batch, hidden_dim]
        transformer_feature = transformer_feature.mean(dim=0)                       # [batch, hidden_dim]
        transformer_feature = F.relu(transformer_feature)
        transformer_feature = self.fc(transformer_feature)                           # [batch, features_dim]

        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # transfer state feature from 0~1 to -1~1
        # state_feature = state_feature*2 - 1

        x = th.cat((transformer_feature, state_feature), dim=1)
        #print("custom_policy_CNN_GAP:x.shape:",x.shape)

        return x
        


