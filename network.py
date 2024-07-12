import torch.nn as nn

class SCN(nn.Module):
    def __init__(self, img_ch, n_landmarks):
        super(SCN, self).__init__()
        self.local_conv1 = nn.Conv2d(img_ch, 64, kernel_size=5, stride=1, padding=2)
        self.local_conv1_1 = nn.Dropout(0.5)
        self.local_conv1_2 = nn.LeakyReLU(0.1, inplace=True)
        self.local_conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.local_conv2_2 = nn.LeakyReLU(0.1, inplace=True)
        self.local_conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.local_conv3_2 = nn.LeakyReLU(0.1, inplace=True)
        self.local_conv4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.local_conv5 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.local_conv5_1 = nn.Dropout(0.5)
        self.local_conv5_2 = nn.LeakyReLU(0.1, inplace=True)
        self.local_conv6 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.local_conv6_2 = nn.LeakyReLU(0.1, inplace=True)
        self.local_conv7 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.local_conv8 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.local_conv8_1 = nn.Dropout(0.5)
        self.local_conv8_2 = nn.LeakyReLU(0.1, inplace=True)
        self.local_conv9 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.local_conv9_2 = nn.LeakyReLU(0.1, inplace=True)

        self.local_conv21 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.local_conv22 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.local_conv22_1 = nn.Dropout(0.5)
        self.local_conv22_2 = nn.LeakyReLU(0.1, inplace=True)
        self.local_conv23 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.local_conv23_2 = nn.LeakyReLU(0.1, inplace=True)
        self.local_conv24 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.local_conv10 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.local_conv12 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.local_conv14 = nn.Conv2d(64, n_landmarks, kernel_size=5, stride=1, padding=2)
        self.local_conv15 = nn.Upsample(scale_factor=1/8, mode='bilinear')#, align_corners=True)
        self.local_conv16 = nn.Conv2d(n_landmarks, 64, kernel_size=15, stride=1, padding=7)
        self.local_conv16_2 = nn.LeakyReLU(0.1, inplace=True)
        self.local_conv17 = nn.Conv2d(64, 64, kernel_size=15, stride=1, padding=7)
        self.local_conv17_2 = nn.LeakyReLU(0.1, inplace=True)
        self.local_conv18 = nn.Conv2d(64, 64, kernel_size=15, stride=1, padding=7)
        self.local_conv18_2 = nn.LeakyReLU(0.1, inplace=True)
        self.local_conv18_5 = nn.Conv2d(64, n_landmarks, kernel_size=15, stride=1, padding=7)
        self.local_conv18_5_2 = nn.Tanh() # after revised as per "the convolution layer generating HSC has a TanH activation function to restrict the outputs between −1 and 1"
        self.local_conv19 = nn.Upsample(scale_factor=8, mode='bicubic')#, align_corners=True)
    
        
        # Weight initialization
        self._initialize_weights()

    def forward(self, x):
        # Local Appearance Component
        x1 = self.local_conv1(x)
        x1 = self.local_conv1_1(x1)
        x1 = self.local_conv1_2(x1)
        
        x2 = self.local_conv2(x1)
        x2 = self.local_conv2_2(x2)
        
        x3 = self.local_conv3(x2)
        x3 = self.local_conv3_2(x3)
        
        x4 = self.local_conv4(x3)
        
        x5 = self.local_conv5(x4)
        x5 = self.local_conv5_1(x5)
        x5 = self.local_conv5_2(x5)
        
        x6 = self.local_conv6(x5)
        x6 = self.local_conv6_2(x6)
        
        x7 = self.local_conv7(x6)
        
        x8 = self.local_conv8(x7)
        x8 = self.local_conv8_1(x8)
        x8 = self.local_conv8_2(x8)
        
        x9 = self.local_conv9(x8)
        x9 = self.local_conv9_2(x9)
        
        x21 = self.local_conv21(x8)
        
        x22 = self.local_conv22(x21)
        x22 = self.local_conv22_1(x22)
        x22 = self.local_conv22_2(x22)

        x23 = self.local_conv23(x22)
        x23 = self.local_conv23_2(x23)

        x24 = self.local_conv24(x23)
        
        x10 = x24 + x9
        x10 = self.local_conv10(x10)
        
        #print(x10.shape)
        #print(x6.shape)
        x11 = x10 + x6
       
        #print(x3.shape)
        x12 = self.local_conv12(x11)
        #print(x12.shape)
        x13 = x12 + x3
        
        x14 = self.local_conv14(x13)
        #print(x14.shape)
        x15 = self.local_conv15(x14)
        
        x16 = self.local_conv16(x15)
        x16 = self.local_conv16_2(x16)
        
        x17 = self.local_conv17(x16)
        x17 = self.local_conv17_2(x17)
        
        x18 = self.local_conv18(x17)
        x18 = self.local_conv18_2(x18)
        
        x18_5 = self.local_conv18_5(x18)
        x18_5 = self.local_conv18_5_2(x18_5)
        
        x19 = self.local_conv19(x18_5)
        #print(x19_2.shape)
        x20 = x19 * x14
        
        return x20

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for Leaky ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
                # Zero initialization for biases
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)