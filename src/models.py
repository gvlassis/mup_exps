import torch

class θNet(torch.nn.Module):
    def __init__(self, θ):
        super(θNet, self).__init__()

        # Convolutional part
        self.layer1 = torch.nn.Conv2d(in_channels=3, out_channels=3*θ, kernel_size=3, stride=2, padding=1)
        self.layer2 = torch.nn.Conv2d(in_channels=3*θ, out_channels=6*θ, kernel_size=3, stride=2, padding=1)
        self.layer3 = torch.nn.Conv2d(in_channels=6*θ, out_channels=12*θ, kernel_size=3, stride=2, padding=1)
        self.layer4 = torch.nn.Conv2d(in_channels=12*θ, out_channels=24*θ, kernel_size=3, stride=2, padding=1)
        self.layer5 = torch.nn.Conv2d(in_channels=24*θ, out_channels=48*θ, kernel_size=3, stride=2, padding=1)
        self.layer6 = torch.nn.Conv2d(in_channels=48*θ, out_channels=96*θ, kernel_size=3, stride=2, padding=1)
        self.layer7 = torch.nn.Conv2d(in_channels=96*θ, out_channels=192*θ, kernel_size=3, stride=2, padding=1)
        self.layer8 = torch.nn.Conv2d(in_channels=192*θ, out_channels=384*θ, kernel_size=3, stride=2, padding=1)

        # Fully connected part
        self.layer9 = torch.nn.Linear(384*θ, 768*θ)
        self.layer10 = torch.nn.Linear(768*θ, 1000)

    def forward(self, x):
        assert (x.shape[-3]==3) and (x.shape[-2]==256) and (x.shape[-1]==256), "Input shaped incorrectly: %dx%dx%d" % (x.shape[-3],x.shape[-2],x.shape[-1])

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

        z7 = self.layer7(a6)
        a7 = torch.nn.functional.relu(z7)

        z8 = self.layer8(a7)
        a8 = torch.nn.functional.relu(z8)

        assert (a8.shape[-3]==384) and (a8.shape[-2]==1) and (a8.shape[-1]==1), "a8 shaped incorrectly: %dx%dx%d" % (a8.shape[-3],a8.shape[-2],a8.shape[-1])

        z9 = self.layer9(a8[...,0,0])
        a9 = torch.nn.functional.relu(z9)

        z10 = self.layer10(a9)
        a10 = torch.nn.functional.softmax(z10,dim=-1)

        assert a10.shape[-1]==1000, "a8 shaped incorrectly: %d" % (a10.shape[-1])

        return a10
