import torch

class θNet_cifar(torch.nn.Module):
    def __init__(self, θ):
        super().__init__()
        # θ can be a string (e.g. if it is passed from the CLI)
        θ = int(θ)

        self.θ = θ

        # Convolutional part
        self.layer1 = torch.nn.Conv2d(in_channels=3, out_channels=12*θ, kernel_size=3, stride=2, padding=1)
        self.layer2 = torch.nn.Conv2d(in_channels=12*θ, out_channels=24*θ, kernel_size=3, stride=2, padding=1)
        self.layer3 = torch.nn.Conv2d(in_channels=24*θ, out_channels=48*θ, kernel_size=3, stride=2, padding=1)
        self.layer4 = torch.nn.Conv2d(in_channels=48*θ, out_channels=96*θ, kernel_size=3, stride=2, padding=1)
        self.layer5 = torch.nn.Conv2d(in_channels=96*θ, out_channels=192*θ, kernel_size=3, stride=2, padding=1)

        # Fully connected part
        self.layer6 = torch.nn.Linear(192*θ, 384*θ)
        self.layer7 = torch.nn.Linear(384*θ, 10)

    def forward(self, x):
        assert (x.shape[-3]==3) and (x.shape[-2]==32) and (x.shape[-1]==32), "Input shaped incorrectly: %d×%d×%d" % (x.shape[-3],x.shape[-2],x.shape[-1])

        z1 = self.layer1(x)
        a1 = torch.nn.functional.relu(z1)

        z2 = self.layer2(a1)
        a2 = torch.nn.functional.relu(z2)

        z3 = self.layer3(a2)
        a3 = torch.nn.functional.relu(z3)

        z4 = self.layer4(a3)
        a4 = torch.nn.functional.relu(z4)

        z5 = self.layer5(a4)
        a5 = torch.nn.functional.relu(z5)

        assert (a5.shape[-3]==192*self.θ) and (a5.shape[-2]==1) and (a5.shape[-1]==1), "a5 shaped incorrectly: %d×%d×%d" % (a5.shape[-3],a5.shape[-2],a5.shape[-1])

        z6 = self.layer6(torch.flatten(a5, start_dim=-3, end_dim=-1))
        a6 = torch.nn.functional.relu(z6)

        z7 = self.layer7(a6)
        a7 = torch.nn.functional.log_softmax(z7,dim=-1)

        assert a7.shape[-1]==10, "a7 shaped incorrectly: %d" % (a7.shape[-1])

        return a7

class θNet_imagenet(torch.nn.Module):
    def __init__(self, θ):
        super().__init__()
        # θ can be a string (e.g. if it is passed from the CLI)
        θ = int(θ)

        self.θ = θ

        # Convolutional part
        self.layer1 = torch.nn.Conv2d(in_channels=3, out_channels=12*θ, kernel_size=3, stride=2, padding=1)
        self.layer2 = torch.nn.Conv2d(in_channels=12*θ, out_channels=24*θ, kernel_size=3, stride=2, padding=1)
        self.layer3 = torch.nn.Conv2d(in_channels=24*θ, out_channels=48*θ, kernel_size=3, stride=2, padding=1)
        self.layer4 = torch.nn.Conv2d(in_channels=48*θ, out_channels=96*θ, kernel_size=3, stride=2, padding=1)
        self.layer5 = torch.nn.Conv2d(in_channels=96*θ, out_channels=192*θ, kernel_size=3, stride=2, padding=1)
        self.layer6 = torch.nn.Conv2d(in_channels=192*θ, out_channels=384*θ, kernel_size=3, stride=2, padding=1)

        # Fully connected part
        self.layer7 = torch.nn.Linear(384*θ, 768*θ)
        self.layer8 = torch.nn.Linear(768*θ, 1000)

    def forward(self, x):
        assert (x.shape[-3]==3) and (x.shape[-2]==64) and (x.shape[-1]==64), "Input shaped incorrectly: %d×%d×%d" % (x.shape[-3],x.shape[-2],x.shape[-1])

        z1 = self.layer1(x)
        a1 = torch.nn.functional.relu(z1)

        z2 = self.layer2(a1)
        a2 = torch.nn.functional.relu(z2)

        z3 = self.layer3(a2)
        a3 = torch.nn.functional.relu(z3)

        z4 = self.layer4(a3)
        a4 = torch.nn.functional.relu(z4)

        z5 = self.layer5(a4)
        a5 = torch.nn.functional.relu(z5)

        z6 = self.layer6(a5)
        a6 = torch.nn.functional.relu(z6)

        assert (a6.shape[-3]==384*self.θ) and (a6.shape[-2]==1) and (a6.shape[-1]==1), "a6 shaped incorrectly: %d×%d×%d" % (a6.shape[-3],a6.shape[-2],a6.shape[-1])

        z7 = self.layer7(torch.flatten(a6, start_dim=-3, end_dim=-1))
        a7 = torch.nn.functional.relu(z7)

        z8 = self.layer8(a7)
        a8 = torch.nn.functional.log_softmax(z8,dim=-1)

        assert a8.shape[-1]==1000, "a8 shaped incorrectly: %d" % (a8.shape[-1])

        return a8

class VGG16_cifar(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional part
        # In: 32x32
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # In: 16x16
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # In: 8x8
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # In: 4x4
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # In: 2x2
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Fully connected part
        # In: 1x1
        self.fully_connected = torch.nn.Sequential(
            torch.nn.Linear(1*1*512, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 10),
            torch.nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        assert (x.shape[-3]==3) and (x.shape[-2]==32) and (x.shape[-1]==32), "Input shaped incorrectly: %d×%d×%d" % (x.shape[-3],x.shape[-2],x.shape[-1])

        a_block1 = self.block1(x)
        a_block2 = self.block2(a_block1)
        a_block3 = self.block3(a_block2)
        a_block4 = self.block4(a_block3)
        a_block5 = self.block5(a_block4)

        assert (a_block5.shape[-3]==512) and (a_block5.shape[-2]==1) and (a_block5.shape[-1]==1), "a_block5 shaped incorrectly: %d×%d×%d" % (a_block5.shape[-3],a_block5.shape[-2],a_block5.shape[-1])

        a_fully_connected = self.fully_connected(torch.flatten(a_block5, start_dim=-3, end_dim=-1))

        assert a_fully_connected.shape[-1]==10, "a_fully_connected shaped incorrectly: %d" % (a_fully_connected.shape[-1])

        return a_fully_connected

class VGG16_imagenet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional part
        # In: 64x64
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # In: 32x32
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # In: 16x16
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # In: 8x8
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        # In: 4x4
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Fully connected part
        # In: 2x2
        self.fully_connected = torch.nn.Sequential(
            torch.nn.Linear(2*2*512, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 1000),
            torch.nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        assert (x.shape[-3]==3) and (x.shape[-2]==64) and (x.shape[-1]==64), "Input shaped incorrectly: %d×%d×%d" % (x.shape[-3],x.shape[-2],x.shape[-1])

        a_block1 = self.block1(x)
        a_block2 = self.block2(a_block1)
        a_block3 = self.block3(a_block2)
        a_block4 = self.block4(a_block3)
        a_block5 = self.block5(a_block4)

        assert (a_block5.shape[-3]==512) and (a_block5.shape[-2]==2) and (a_block5.shape[-1]==2), "a_block5 shaped incorrectly: %d×%d×%d" % (a_block5.shape[-3],a_block5.shape[-2],a_block5.shape[-1])

        a_fully_connected = self.fully_connected(torch.flatten(a_block5, start_dim=-3, end_dim=-1))

        assert a_fully_connected.shape[-1]==1000, "a_fully_connected shaped incorrectly: %d" % (a_fully_connected.shape[-1])

        return a_fully_connected

class transformer_block(torch.nn.Module):
    def __init__(self, D, heads, hidden):
        super().__init__()

        self.D = D
        self.heads = heads
        self.hidden = hidden

        self.norm1 = torch.nn.LayerNorm(D)
        self.attention = torch.nn.MultiheadAttention(D, heads)
        self.norm2 = torch.nn.LayerNorm(D)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(D,hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden,D)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        y = self.attention(x_norm,x_norm,x_norm)[0] + x
        z = self.MLP(self.norm2(y)) + y

        return z

class θViT_cifar(torch.nn.Module):
    def __init__(self, P, L, heads, θ):
        super().__init__()
        # P, L, heads, θ can be a strings (e.g. if they are passed from the CLI)
        P = int(P)
        L = int(L)
        heads = int(heads)
        θ = int(θ)

        if 32 % P != 0:
            raise ValueError("P (Patch size)=%d should be a divisor of 32" % P)

        self.P = P
        self.L = L
        self.heads = heads
        self.θ = θ

        self.N = int(32/P)*int(32/P)
        self.D = 48*θ
        self.hidden = 2*self.D

        self.embedding_layer = torch.nn.Conv2d(in_channels=3, out_channels=self.D, kernel_size=P, stride=P)

        self.class_token = torch.nn.Parameter(torch.empty((self.D)))

        self.pos_embeddings = torch.nn.Parameter(torch.empty((self.N+1, self.D)))

        self.transformer_blocks = torch.nn.Sequential(*[transformer_block(self.D, heads, self.hidden) for _ in range(L)])

        self.linear_layer = torch.nn.Linear(self.D, 10)

    def forward(self, x):
        assert (x.shape[-3]==3) and (x.shape[-2]==32) and (x.shape[-1]==32), "Input shaped incorrectly: %d×%d×%d" % (x.shape[-3],x.shape[-2],x.shape[-1])

        embeddings = torch.swapdims(torch.flatten(self.embedding_layer(x),start_dim=-2),-2,-1)
        # embeddings.shape = (batch_size*)N*D

        # x: single
        if len(x.shape)==3:
            x_ = torch.cat((self.class_token[None,:], embeddings), dim=0)
        # x: batch
        else:
            x_ = torch.cat((self.class_token.repeat(x.shape[0],1,1), embeddings), dim=1)

        # x_.shape = (batch_size*)(N+1)*D

        y = self.transformer_blocks(x_)

        assert (y.shape[-2]==self.N+1) and (y.shape[-1]==self.D), "y shaped incorrectly: %d×%d" + (y.shape[-2], y.shape[-1])

        y0 = y[...,0,:]

        z0 = self.linear_layer(y0)
        a0 = torch.nn.functional.log_softmax(z0,dim=-1)

        assert a0.shape[-1]==10, "a0 shaped incorrectly: %d" % (a0.shape[-1])

        return a0