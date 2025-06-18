from chainer import Sequential
import chainer.links as L
import chainer.functions as F
import chainer

def conv_bn_relu(kernel_size, input_dim, output_dim, stride=1, with_bn=True) -> Sequential:
    pad = (kernel_size - 1) // 2
    bn = L.BatchNormalization(output_dim) if with_bn else F.identity
    return Sequential(
        L.Convolution2D(input_dim, output_dim, kernel_size, stride, pad, nobias=(not with_bn)),
        bn,
        F.relu,
    )

class BasicBlock(chainer.Chain):
    def __init__(self, n_in, n_mid, n_out, stride=1, proj=False,initialW=chainer.initializers.HeNormal()):
        super(BasicBlock, self).__init__()
        with self.init_scope():
            self.conv3x3a = L.Convolution2D(n_in, n_mid, 3, stride, 1, initialW=initialW, nobias=True)
            self.conv3x3b = L.Convolution2D(n_mid, n_out, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn_a = L.BatchNormalization(n_mid)
            self.bn_b = L.BatchNormalization(n_out)
            if proj:
                self.conv1x1r = L.Convolution2D(n_in, n_out, 1, stride, 0, initialW=initialW, nobias=True)
                self.bn_r = L.BatchNormalization(n_out)
        self.proj = proj

    def __call__(self, x):
        h = F.relu(self.bn_a(self.conv3x3a(x)))
        h = self.bn_b(self.conv3x3b(h))
        if self.proj:
            x = self.bn_r(self.conv1x1r(x))
        return F.relu(h + x)

class BottleNeck(chainer.Chain):
    def __init__(self, n_in, n_mid, n_out, stride=1, proj=False,initialW=chainer.initializers.HeNormal()):
        super(BottleNeck, self).__init__()
        with self.init_scope():
            self.conv1x1a = L.Convolution2D(n_in, n_mid, 1, stride, 0, initialW=initialW, nobias=True)
            self.conv3x3b = L.Convolution2D(n_mid, n_mid, 3, 1, 1, initialW=initialW, nobias=True)
            self.conv1x1c = L.Convolution2D(n_mid, n_out, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn_a = L.BatchNormalization(n_mid)
            self.bn_b = L.BatchNormalization(n_mid)
            self.bn_c = L.BatchNormalization(n_out)
            if proj:
                self.conv1x1r = L.Convolution2D(n_in, n_out, 1, stride, 0, initialW=initialW, nobias=True)
                self.bn_r = L.BatchNormalization(n_out)
        self.proj = proj

    def __call__(self, x):
        h = F.relu(self.bn_a(self.conv1x1a(x)))
        h = F.relu(self.bn_b(self.conv3x3b(h)))
        h = self.bn_c(self.conv1x1c(h))
        if self.proj:
            x = self.bn_r(self.conv1x1r(x))
        return F.relu(h + x)

class ResBlock(chainer.ChainList):
    def __init__(self, block, n_layers, n_in, n_mid, n_out, stride=2):
        super(ResBlock, self).__init__()
        self.add_link(block(n_in, n_mid, n_out, stride, True))
        for _ in range(n_layers - 1):
            self.add_link(block(n_out, n_mid, n_out))
        self.out_size = n_out

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x
    
class ResNet_CenterNet(chainer.Chain):
    cfgs={
        'resnet18':{'block':BasicBlock, 'blocks_num':[2, 2, 2, 2],'expansion':1},
        'resnet34':{'block':BasicBlock, 'blocks_num':[3, 4, 6, 3],'expansion':1},
        'resnet50':{'block':BottleNeck, 'blocks_num':[3, 4, 6, 3],'expansion':4},
        'resnet101':{'block':BottleNeck, 'blocks_num':[3, 4, 23, 3],'expansion':4},
        'resnet152':{'block':BottleNeck, 'blocks_num':[3, 8, 36, 3],'expansion':4},
    }
    def __init__(self,model_name='resnet18',channels=3,heads=None,alpha=1.0,initialW=chainer.initializers.HeNormal(),roi_size=7):
        super(ResNet_CenterNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=channels, out_channels = int(64*alpha), ksize=7, stride=2, pad=3, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(int(64*alpha))
                
            self.res2 = ResBlock(self.cfgs[model_name]['block'],self.cfgs[model_name]['blocks_num'][0], int(64*alpha), int(64*alpha), int(64*alpha)*self.cfgs[model_name]['expansion'], 1)
            self.res3 = ResBlock(self.cfgs[model_name]['block'],self.cfgs[model_name]['blocks_num'][1], int(64*alpha)*self.cfgs[model_name]['expansion'], int(128*alpha), int(128*alpha)*self.cfgs[model_name]['expansion'],1)
            self.res4 = ResBlock(self.cfgs[model_name]['block'],self.cfgs[model_name]['blocks_num'][2], int(128*alpha)*self.cfgs[model_name]['expansion'], int(256*alpha), int(256*alpha)*self.cfgs[model_name]['expansion'],1)
            self.out5 = ResBlock(self.cfgs[model_name]['block'],self.cfgs[model_name]['blocks_num'][3], int(256*alpha)*self.cfgs[model_name]['expansion'], int(512*alpha), int(512*alpha)*self.cfgs[model_name]['expansion'],1)

            for head in heads:
                self.__setattr__(head, conv_bn_relu(3, int(512*alpha)*self.cfgs[model_name]['expansion'], heads[head]))
            self.heads = heads


    def __call__(self, x):
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 2, 2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.out5(h)
        
        out = {}

        for head in self.heads:
            layer = self.__getattribute__(head)
            y = layer(h)
            out[head] = y

        return [out]