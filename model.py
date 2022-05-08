from util import *
from collections import OrderedDict

random_seed=0
torch.manual_seed(random_seed)


class FFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, bicls = False):
        super(FFNN, self).__init__()
        self.input_size = input_size
        self.bicls = bicls
        dic = OrderedDict()
        previous_dim = input_size
        for i,dim in enumerate(hidden_sizes):
            lay = nn.Linear(previous_dim,dim)
            previous_dim = dim
            dic['linear'+str(i)]=lay
            dic['act_func'+str(i)]=nn.ReLU()
                
        n_hid = len(hidden_sizes)        
        if bicls:
            lay = nn.Linear(previous_dim,1)
        else:
            lay = nn.Linear(previous_dim,num_classes)
        dic['linear'+str(n_hid)]=lay
        self.seq = nn.Sequential(dic)
        
        
    def forward(self, x):
        out = self.seq(x)
        return out 

        
    

class IFFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, bicls = False,use_dropout=False, act_func = 'relu'):
        super(IFFNN, self).__init__()
        self.input_size = input_size
        self.bicls = bicls
        self.num_classes = num_classes
        dic = OrderedDict()
        previous_dim = input_size
        for i,dim in enumerate(hidden_sizes):
            lay = nn.Linear(previous_dim,dim)
            previous_dim = dim
            dic['linear'+str(i)]=lay
            if act_func == 'tanh':
                dic['act_func'+str(i)]=nn.Tanh()
            else:
                assert(act_func == 'relu')
                dic['act_func'+str(i)]=nn.ReLU()
                
        n_hid = len(hidden_sizes)        
        if bicls:
            lay = nn.Linear(previous_dim,input_size)
            self.last_bias = torch.nn.Parameter(torch.zeros([1]))
        else:
            lay = nn.Linear(previous_dim,input_size*num_classes)
            self.last_bias = torch.nn.Parameter(torch.zeros([num_classes]))
            
        dic['linear'+str(n_hid)]=lay
        self.iffnnpart1 = nn.Sequential(dic)
        
        self.register_parameter(name='bias', param=self.last_bias)
        
        
    def forward(self, x):
        out = self.iffnnpart1(x)
        if self.bicls:
            full_features = x
        else:
            full_features = x.repeat(1,self.num_classes)
        out = full_features*out
        if self.bicls:
            out = out.sum(axis=1)
        else:
            out = out.reshape(-1,self.num_classes,self.input_size)
            out = out.sum(axis=2)
        out = out + self.last_bias
        return out 
    def explain(self, x):
        out = self.iffnnpart1(x)
        
        if self.bicls:
            full_features = x
        else:
            full_features = x.repeat(1,self.num_classes)
        out = full_features*out
        if not self.bicls:
            out = out.reshape(-1,self.num_classes,self.input_size)
        out = out.cpu().detach().numpy()    
        x = x.cpu().detach().numpy()
        if self.bicls:
            results = explain_binary(x,out)
        else:
            results = explain_multi(x,out,self.num_classes)
        
        return results 

class SoftmaxRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, num_classes) 
        self.num_classes = num_classes
    def forward(self, x):
        out = self.l1(x)
        return out 
    def explain(self, x):
        out = (x.repeat(1,self.num_classes).reshape(x.shape[0],self.num_classes,self.input_size)*(self.l1.weight))
        out = out.cpu().detach().numpy()
        x = x.cpu().detach().numpy()
        results = explain_multi(x,out,self.num_classes)
        return results
        
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, 1) 
    def forward(self, x):
        out = self.l1(x)
        return out 
    def explain(self, x):
        out = x*self.l1.weight
        out = out.cpu().detach().numpy()
        x = x.cpu().detach().numpy()
        results = explain_binary(x,out)
        return results
        
class CNNNet(nn.Module):
    def __init__(self,num_classes = 10, bicls = False):
        super(CNNNet, self).__init__()
        self.num_classes = num_classes
        self.bicls = bicls
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)   
        if bicls:
            self.fc2 = nn.Linear(128, 1)
        else:
            self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class I_CNNNet(nn.Module):
    def __init__(self,num_classes=10, bicls = False):
        super(I_CNNNet, self).__init__()
        self.num_classes = num_classes
        self.bicls = bicls
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        if bicls:
            self.fc1 = nn.Linear(9216, 28*28)
            self.last_bias = torch.nn.Parameter(torch.zeros([1]))
        else:
            self.fc1 = nn.Linear(9216, 28*28*num_classes)
            self.last_bias = torch.nn.Parameter(torch.zeros([num_classes]))
        self.register_parameter(name='bias', param=self.last_bias)
    def forward(self, x):
        x0 = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        if self.bicls:
            out = x*(x0.reshape(-1,784))
            out = out.sum(axis=1)+self.last_bias
        
        else:
            x = x*(x0.reshape(-1,784).repeat(1,self.num_classes))
            out = x.reshape(-1,self.num_classes,784)
            out = out.sum(axis=2)+self.last_bias
        return out
    def explain(self, x):
        x0 = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        if self.bicls:
            out = x*(x0.reshape(-1,784))
        else:
            x = x*(x0.reshape(-1,784).repeat(1,self.num_classes))
            out = x.reshape(-1,self.num_classes,784)
        out = out.cpu().detach().numpy()    
        x=x0.reshape(-1,784).cpu().detach().numpy()
        
        if self.bicls:
            results = explain_binary(x,out)
        else:
            results = explain_multi(x,out,self.num_classes)
        return results 

class CNNNet2(nn.Module):
    def __init__(self,num_classes = 10, bicls = False):
        super(CNNNet2, self).__init__()
        self.num_classes = num_classes
        self.bicls = bicls
        self.conv1 = nn.Conv2d(1, 512, 3, 1)
        self.conv2 = nn.Conv2d(512, 870, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(125280, 545)
        if bicls:
            self.fc2 = nn.Linear(545, 1)
        else:
            self.fc2 = nn.Linear(545, num_classes)
        

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
class CNNNet_BC2(nn.Module):
    def __init__(self,num_classes = 2, bicls = True):
        super(CNNNet_BC2, self).__init__()
        self.num_classes = num_classes
        self.bicls = True
        self.conv1 = nn.Conv2d(1, 532, 3, 1)
        self.conv2 = nn.Conv2d(532, 100, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(14400, 468)
        if bicls:
            self.fc2 = nn.Linear(468, 1)
        else:
            self.fc2 = nn.Linear(468, num_classes)
        

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
class CNNNet2_MC2(nn.Module):
    def __init__(self,num_classes = 2, bicls = False):
        super(CNNNet2_MC2, self).__init__()
        self.num_classes = num_classes
        self.bicls = False
        self.conv1 = nn.Conv2d(1, 532, 3, 1)
        self.conv2 = nn.Conv2d(532, 100, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(14400, 970)
        if bicls:
            self.fc2 = nn.Linear(970, 1)
        else:
            self.fc2 = nn.Linear(970, num_classes)
        

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
class CNNNet3(nn.Module):
    def __init__(self,num_classes = 10, bicls = False):
        super(CNNNet3, self).__init__()
        self.bicls = bicls
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(30976, 2238)
        self.fc2 = nn.Linear(2238, 1024)
        self.fc3 = nn.Linear(1024, 256)
        if bicls:
            self.fc4 = nn.Linear(256, 1)
        else:
            self.fc4 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        x = self.dropout4(x)
        x = self.fc4(x)
        #output = F.log_softmax(x, dim=1)
        return x

class CNNNet_BC3(nn.Module):
    def __init__(self,num_classes = 2, bicls = False):
        super(CNNNet_BC3, self).__init__()
        self.bicls = bicls
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 72, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(8712, 768)
        self.fc2 = nn.Linear(768, 472)
        self.fc3 = nn.Linear(472, 256)
        if bicls:
            self.fc4 = nn.Linear(256, 1)
        else:
            self.fc4 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        x = self.dropout4(x)
        x = self.fc4(x)
        return x
        
        
class CNNNet2_MC3(nn.Module):
    def __init__(self,num_classes = 2, bicls = False):
        super(CNNNet2_MC3, self).__init__()
        self.bicls = False
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 132, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(15972, 865)
        self.fc2 = nn.Linear(865, 512)
        self.fc3 = nn.Linear(512, 256)
        if bicls:
            self.fc4 = nn.Linear(256, 1)
        else:
            self.fc4 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        x = self.dropout4(x)
        x = self.fc4(x)
        return x
        
class HighwayLayer(nn.Module):
    def __init__(self, input_size, bias=-2):
        super(HighwayLayer, self).__init__()
        self.fc = nn.Linear(input_size, input_size)
        self.gate = nn.Linear(input_size, input_size)
        self.gate.bias.data.fill_(bias)

    def forward(self, x):
        H=self.fc(x)
        T=self.gate(x)
        out = H*T+x*(1-T)
        return out

class HWNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, bicls = False):
        super(HWNet, self).__init__()
        self.input_size = input_size
        self.bicls = bicls
        dic = OrderedDict()
        previous_dim = input_size
        for i,dim in enumerate(hidden_sizes):
            lay = nn.Linear(previous_dim,dim)
            dic['linear'+str(i)]=lay
            dic['act_func'+str(i)]=nn.ReLU()
            hw = HighwayLayer(dim)
            dic['hw'+str(i)+'_hw']=hw
            dic['act_func'+str(i)+'_hw']=nn.ReLU()
            previous_dim = dim
        n_hid = len(hidden_sizes)        
        if bicls:
            lay = nn.Linear(previous_dim,1)
        else:
            lay = nn.Linear(previous_dim,num_classes)
        
        dic['linear'+str(n_hid)]=lay
        self.seq = nn.Sequential(dic)
    def forward(self, x):
        out = self.seq(x)
        return out 

class HW_IFFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, bicls = False):
        super(HW_IFFNN, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.bicls = bicls
        dic = OrderedDict()
        previous_dim = input_size
        for i,dim in enumerate(hidden_sizes):
            lay = nn.Linear(previous_dim,dim)
            dic['linear'+str(i)]=lay
            dic['act_func'+str(i)]=nn.ReLU()
            hw = HighwayLayer(dim)
            dic['hw'+str(i)+'_hw']=hw
            dic['act_func'+str(i)+'_hw']=nn.ReLU()
            previous_dim = dim
        n_hid = len(hidden_sizes)        
        if bicls:
            lay = nn.Linear(previous_dim,self.input_size)
            self.last_bias = torch.nn.Parameter(torch.zeros([1]))
        else:
            lay = nn.Linear(previous_dim,self.input_size*num_classes)
            self.last_bias = torch.nn.Parameter(torch.zeros([num_classes]))
        dic['linear'+str(n_hid)]=lay
        self.seq = nn.Sequential(dic)
        self.register_parameter(name='bias', param=self.last_bias)
    def forward(self, x):
        out = self.seq(x)
        if self.bicls:
            full_features = x
        else:
            full_features = x.repeat(1,self.num_classes)
        out = full_features*out
        if self.bicls:
            out = out.sum(axis=1)
        else:
            out = out.reshape(-1,self.num_classes,self.input_size)
            out = out.sum(axis=2)
        out = out + self.last_bias
        return out 
    def explain(self, x):
        out = self.seq(x)
        if self.bicls:
            full_features = x
        else:
            full_features = x.repeat(1,self.num_classes)
        out = full_features*out
        if not self.bicls:
            out = out.reshape(-1,self.num_classes,self.input_size)
        out = out.cpu().detach().numpy()
        x = x.cpu().detach().numpy()
        if self.bicls:
            results = explain_binary(x,out)
        else:
            results = explain_multi(x,out,self.num_classes)
        return results 
    

#======================================================================
#All the following residual-network-related functions and classes are modified from the tutorial https://www.kaggle.com/readilen/resnet-for-mnist-with-pytorch
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, bicls = False):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.bicls = bicls
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        if bicls:
            self.fc = nn.Linear(64*7*7, 1)
        else:
            self.fc = nn.Linear(64*7*7, num_classes)
        

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

class ResNet_IFFNN(nn.Module):
    def __init__(self, block, layers, num_classes=10, bicls = False):
        super(ResNet_IFFNN, self).__init__()
        self.num_classes = num_classes
        self.bicls = bicls
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        if bicls:
            self.fc = nn.Linear(64*7*7, 28*28)
            self.last_bias = torch.nn.Parameter(torch.zeros([1]))
        else:
            self.fc = nn.Linear(64*7*7, 28*28*self.num_classes)
            self.last_bias = torch.nn.Parameter(torch.zeros([self.num_classes]))
        
        self.register_parameter(name='bias', param=self.last_bias)
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.bicls:
            out = x0.reshape(-1,784)*x
            out = out.sum(axis=1)+self.last_bias
        
        else:
            x = x*(x0.reshape(-1,784).repeat(1,self.num_classes))
            out = x.reshape(-1,self.num_classes,784)
            out = out.sum(axis=2)+self.last_bias
        return out
        
    def explain(self, x):
        x0 = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.bicls:
            out = x0.reshape(-1,784)*x
        else:
            x = x*(x0.reshape(-1,784).repeat(1,self.num_classes))
            out = x.reshape(-1,self.num_classes,784)
        out = out.cpu().detach().numpy()
        x=x0.reshape(-1,784).cpu().detach().numpy()
        if self.bicls:
            results = explain_binary(x,out)
        else:
            results = explain_multi(x,out,self.num_classes)
        return results 
class ResNet2(nn.Module):
    def __init__(self, block, layers, num_classes=10, bicls = False):
        super(ResNet2, self).__init__()
        self.in_channels = 16
        self.bicls = bicls
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 160, layers[0])
        self.layer2 = self.make_layer(block, 360, layers[0], 2)
        self.layer3 = self.make_layer(block, 729, layers[1], 2)
        if bicls:
            self.fc = nn.Linear(729*7*7, 1)
        else:
            self.fc = nn.Linear(729*7*7, num_classes)
        

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
class ResNet_BC2(nn.Module):
    def __init__(self, block, layers, num_classes=10, bicls = False):
        super(ResNet_BC2, self).__init__()
        self.in_channels = 16
        self.bicls = bicls
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[0], 2)
        self.layer3 = self.make_layer(block, 228, layers[1], 2)
        if bicls:
            self.fc = nn.Linear(228*7*7, 1)
        else:
            self.fc = nn.Linear(228*7*7, num_classes)
        

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
        
class ResNet2_MC2(nn.Module):
    def __init__(self, block, layers, num_classes=2, bicls = False):
        super(ResNet2_MC2, self).__init__()
        self.in_channels = 16
        self.bicls = bicls
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 128, layers[0])
        self.layer2 = self.make_layer(block, 220, layers[0], 2)
        self.layer3 = self.make_layer(block, 256, layers[1], 2)
        if bicls:
            self.fc = nn.Linear(256*7*7, 1)
        else:
            self.fc = nn.Linear(256*7*7, num_classes)
        

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
class ResNet3(nn.Module):
    def __init__(self, block, layers, num_classes=10, bicls = False):
        super(ResNet3, self).__init__()
        self.in_channels = 16
        self.bicls = bicls
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[0], 2)
        self.layer3 = self.make_layer(block, 256, layers[1], 2)
        self.layer4 = self.make_layer(block, 512, layers[1], 2)
        self.fc1 = nn.Linear(512*4*4, 1324)
        self.fc2 = nn.Linear(1324, 1024)
        if bicls:
            self.fc3 = nn.Linear(1024, 1)
        else:
            self.fc3 = nn.Linear(1024, num_classes)


    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.fc3(x)
        return out       
class ResNet_BC3(nn.Module):
    def __init__(self, block, layers, num_classes=10, bicls = False):
        super(ResNet_BC3, self).__init__()
        self.in_channels = 16
        self.bicls = bicls
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 32, layers[0])
        self.layer2 = self.make_layer(block, 64, layers[0], 2)
        self.layer3 = self.make_layer(block, 96, layers[1], 2)
        self.layer4 = self.make_layer(block, 140, layers[1], 2)
        self.fc1 = nn.Linear(140*4*4, 536)
        self.fc2 = nn.Linear(536, 256)
        if bicls:
            self.fc3 = nn.Linear(256, 1)
        else:
            self.fc3 = nn.Linear(256, num_classes)


    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.fc3(x)
        return out       
        

class ResNet2_MC3(nn.Module):
    def __init__(self, block, layers, num_classes=10, bicls = False):
        super(ResNet2_MC3, self).__init__()
        self.in_channels = 16
        self.bicls = bicls
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 32, layers[0])
        self.layer2 = self.make_layer(block, 64, layers[0], 2)
        self.layer3 = self.make_layer(block, 96, layers[1], 2)
        self.layer4 = self.make_layer(block, 256, layers[1], 2)
        self.fc1 = nn.Linear(256*4*4, 540)
        self.fc2 = nn.Linear(540, 256)
        if bicls:
            self.fc3 = nn.Linear(256, 1)
        else:
            self.fc3 = nn.Linear(256, num_classes)


    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.fc3(x)
        return out       
#======================================================================