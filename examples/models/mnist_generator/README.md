
# **MNIST Generator Model**  

## **Purpose**  
Demonstrates how to convert a `.pth` model into a `.pt` (TorchScript) model compatible with Triton.  

## **Model Source**  
The original model can be found here:  
[GitHub - GAN/VAE Pretrained PyTorch](https://github.com/csinva/gan-vae-pretrained-pytorch/blob/master/mnist_dcgan/weights/netG_epoch_99.pth).  

## **Conversion Process**  

### **Step 1: Identify the Model Architecture**  
Locate the model definition (`__init__` and `forward()` methods). For this example, the architecture is defined [here](https://github.com/csinva/gan-vae-pretrained-pytorch/blob/master/mnist_dcgan/dcgan.py#L19-L49):  
```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, nc=1, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
```

### **Step 2: Convert `.pth` to `.pt`**  
Use the following script to load and convert the model:  
```python
import torch

# Model initialization
num_gpu = 1 if torch.cuda.is_available() else 0
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
G = Generator(ngpu=num_gpu).eval()

# Load the weights
G.load_state_dict(torch.load('./netG_epoch_99.pth', map_location=device))

# Convert and save to TorchScript
example_input = torch.randn(1, 100, 1, 1)
traced_script_module = torch.jit.trace(G, example_input)
traced_script_module.save("./Generator.pt")
```

### **Step 3: Move the Model**  
Transfer the generated `.pt` file to the model folder:  
```sh
$ mv Generator.pt generator/1/model.pt
```  