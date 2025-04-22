import torch
import torch.nn as nn



class Net(nn.Module) :
  def __init__(self) :
    super().__init__()
    self.conv1 = nn.Conv2d(1,32,3,padding='same')
    self.activ1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(2)

    self.conv2 = nn.Conv2d(32,64,3,padding='same')
    self.activ2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(2)

    self.conv3 = nn.Conv2d(64,128,3,padding='same')
    self.activ3 = nn.ReLU()
    self.pool3 = nn.MaxPool2d(2)

    self.conv4 = nn.Conv2d(128,256,3,padding='same')
    self.activ4 = nn.ReLU()
    self.pool4 = nn.MaxPool2d(2)

    self.flatten = nn.Flatten()

    self.dense1 = nn.Linear(256 * 8 * 8, 256)
    self.dropout1 = nn.Dropout(p=0.5)
    self.activ5 = nn.ReLU()

    self.dense2 = nn.Linear(256, 128)
    self.dropout2 = nn.Dropout(p=0.5)
    self.activ6 = nn.ReLU()

    self.dense3 = nn.Linear(128, 1)
    self.sigmoid = nn.Sigmoid()



  def forward(self, x) : # dimension de x : (128,128), en général je met les dimensions en sortie de couche après la couche.
    x = self.conv1(x)
    #taille 32,128,128
    x = self.activ1(x)
    x = self.pool1(x)
    #taille 32,64,64

    x = self.conv2(x)
    #taille 64,64,64
    x = self.activ2(x)
    x = self.pool2(x)
    #taille 64,32,32

    x = self.conv3(x)
    #taille 128,32,32
    x = self.activ3(x)
    x = self.pool3(x)
    #taille 128,16,16

    x = self.conv4(x)
    #taille 256,16,16
    x = self.activ4(x)
    x = self.pool4(x)
    #taille 256,8,8

    x = self.flatten(x)
    #taille 256 * 8 * 8

    x = self.dense1(x)
    #taille 256
    x = self.dropout1(x)
    x = self.activ5(x)

    x = self.dense2(x)
    #taille 128
    x = self.dropout2(x)
    x = self.activ6(x)

    x = self.dense3(x)
    #taille 1
    x = self.sigmoid(x)

    return x.squeeze() # le squeeze enlève tous les 1 des dimensions