### Pix2Pix-based Signature Image Conversion
`2024.12.1`
This is my first blog post, documenting my first experience using PyTorch to complete an AI-generated signature image conversion: The principle of this model is to first use `font_generate.py` to generate a Song font image, and then use the trained model to convert this Song font image into a signature. Due to limited technical ability, I can only use this crude method, but I will definitely try diffusion fine-tuning in the future!

[Here is the project address](https://aistudio.baidu.com/projectdetail/8576777), I have not put it on GitHub for full open source...

----
##### Project File Distribution:
```
- fonts/
    - 仿宋.ttf
    ...
- generated_image/
    - XXX.png
    ...
- targeted_image/
    - XXX.png
    ...
- font_generate.py
- rename_data.py
- Pix2pixmodel.py
```

#### 1. Dataset Preparation:

I had several enthusiastic freshman students prepare the dataset for me. Each student prepared 300 signature datasets for me, all in PNG format, with the file name being the name signed in the image. After organizing them, I am ready to place them in my folder for processing.

The correct method is to pack these images into the `targeted_img` folder, which will be read and then trained.

#### 2. Dataset Preprocessing:
`font_generate.py` is used to generate Song font images. I used `font_generate.py` to generate an image, the name of which is the name signed. By reading the image names in the `targeted_image` folder and putting them into the program, I can batch generate Song images corresponding to the names of these images, which will be stored in the `generated_image` folder.

The source code of `font_generate.py` is as follows:

```python
from PIL import Image, ImageDraw, ImageFont
import os
import re

def generate_img(text, font_url, output_url):
    img = Image.new('RGB', (500, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_size = int(0.6 * img.height)
    font = ImageFont.truetype(font_url, size=font_size)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (img.width - text_width) // 2
    y = (img.height - text_height) // 2
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    img.save(output_url)

folder_path = './targeted_image/'
file_names = os.listdir(folder_path)

for file in file_names:
    match = re.match(r'(.+)\.png$', file)
    if match:
        text = match.group(1)
        output_url = f'./generated_image/{text}.png' 
        font_path = './fonts/FangZhengFangSong-GBK-1.ttf'
        generate_img(text, font_path, output_url)
```

#### 3. Further Dataset Preprocessing
This preprocessing doesn't seem so easy to write, so I divided it into two steps. First, I batch renamed all the images used for training, but batch renaming can be confusing, so I used a name mapping to ensure that images with the same name are renamed to the same numbered name, so as not to let the effect be too bad. After running `rename_data.py`, all image names will be changed to numbered names, which actually benefits my model training:

The source code of `rename_data.py` is as follows:

```python
import os

filename = os.listdir('./generated_image/')
number = list(range(1,153))
filename_list = [str(i) + '.png' for i in number]
name_map = dict(zip(filename,filename_list))

def rename_files_in_folders(folder_a, name_mapping):
    # Traverse files in folder A
    for filename in os.listdir(folder_a):
        if filename in name_mapping:
            old_file_path = os.path.join(folder_a, filename)
            new_file_path = os.path.join(folder_a, name_mapping[filename])
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {old_file_path} -> {new_file_path}')
            
input_img_path = './generated_image/'
target_img_path = './targeted_image/'
            
rename_files_in_folders(input_img_path,name_map)
rename_files_in_folders(target_img_path,name_map)
```

#### 4. Model Building and Training:

I'm not very good at advanced models, so I chose the Pix2Pix model as a second best. The model building and training code are also benchmarks I found online.

##### 4.1 Dataset Establishment:

First, import the necessary libraries:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision # Load images
from torchvision import transforms # Image transformations
 
import numpy as np
import matplotlib.pyplot as plt # Plotting
import os
```

Then use `dataloader` to establish the dataset:

```python
imgs_path = glob.glob('generated_image/*.png')
annos_path = glob.glob('targeted_image/*.png')

imgs_path.sort()
annos_path.sort()

# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256,256)),
    transforms.Normalize(mean=0.5,std=0.5)
])
 
# Define dataset
class CMP_dataset(data.Dataset):
    def __init__(self,imgs_path,annos_path):
        self.imgs_path =imgs_path
        self.annos_path = annos_path
    def __getitem__(self,index): 
        img_path = self.imgs_path[index]
        anno_path = self.annos_path[index]
        pil_img = Image.open(img_path) # Read data
        pil_img = transform(pil_img)  # Transform data
        anno_img = Image.open(anno_path) # Read data
        anno_img = anno_img.convert("RGB")
        pil_anno = transform(anno_img)  # Transform data
        return pil_anno,pil_img
    def __len__(self):
        return len(self.imgs_path)

# Create dataset
dataset = CMP_dataset(imgs_path,annos_path)

# Convert data to dataloader format for easy iteration
BATCHSIZE = 32
dataloader = data.DataLoader(dataset,
                            batch_size = BATCHSIZE,
                            shuffle = True)
annos_batch,imgs_batch = next(iter(dataloader))
```
##### 4.2 Building the Model:

- **Generator**
The generator consists of 6 downsampling modules (Downsample), 5 upsampling modules (Upsample), and an output layer. Each downsampling module includes a convolutional layer (Conv2d), a LeakyReLU activation function, and a batch normalization layer (BatchNorm2d). The upsampling modules use transposed convolutional layers (ConvTranspose2d) for upsampling and also include LeakyReLU activation functions and batch normalization layers. Some upsampling modules also contain Dropout layers to prevent overfitting. The final output of the generator is achieved through a transposed convolutional layer that converts feature maps into RGB images.
- **Discriminator**
The discriminator is composed of 2 downsampling modules and an additional convolutional layer. Similar to the generator, the downsampling modules contain convolutional layers, LeakyReLU activation functions, and batch normalization layers. The input to the discriminator is the concatenation of the annotation image and the image to be discriminated, and the output is a single-channel discrimination result, indicating the probability that the input image pair is real.
- **Hyperparameter Settings**
•	Learning Rate: The learning rate for both the generator and discriminator is set to 1e-3.
•	Optimizer: Adam optimizer is used with betas set to (0.5, 0.999).
•	Batch Size: BATCHSIZE is set to 32.
•	Loss Function: Binary cross-entropy loss (BCELoss) is used to calculate the discriminator's loss, while the generator's loss includes adversarial loss (also using BCELoss) and L1 loss, with the weight of L1 loss, LAMBDA, set to 7.
•	Number of Training Epochs: num_epoch is set to 100.

```python
class Downsample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Downsample,self).__init__()
        self.conv_relu = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,
                 kernel_size=3,
                 stride=2,
                 padding=1),
        nn.LeakyReLU(inplace=True))
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self,x,is_bn=True):
        x=self.conv_relu(x)
        if is_bn:
            x=self.bn(x)
        return x

class Upsample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Upsample,self).__init__()
        self.upconv_relu = nn.Sequential(
        nn.ConvTranspose2d(in_channels,out_channels,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 output_padding=1),
        nn.LeakyReLU(inplace=True))
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self,x,is_drop=False):
        x=self.upconv_relu(x)
        x=self.bn(x)
        if is_drop:
            x=F.dropout2d(x)
        return x
  
# Generator: 6 downsampling, 5 upsampling, one output layer
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.down1 = Downsample(3,64)      #64,128,128
        self.down2 = Downsample(64,128)    #128,64,64
        self.down3 = Downsample(128,256)   #256,32,32
        self.down4 = Downsample(256,512)   #512,16,16
        self.down5 = Downsample(512,512)   #512,8,8
        self.down6 = Downsample(512,512)   #512,4,4
        
        self.up1 = Upsample(512,512)    #512,8,8
        self.up2 = Upsample(1024,512)   #512,16,16
        self.up3 = Upsample(1024,256)   #256,32,32
        self.up4 = Upsample(512,128)    #128,64,64
        self.up5 = Upsample(256,64)     #64,128,128
        
        self.last = nn.ConvTranspose2d(128,3,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1)  #3,256,256
        
    def forward(self,x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        
        x6 = self.up1(x6,is_drop=True)
        x6 = torch.cat([x6,x5],dim=1)
        
        x6 = self.up2(x6,is_drop=True)
        x6 = torch.cat([x6,x4],dim=1)
        
        x6 = self.up3(x6,is_drop=True)
        x6 = torch.cat([x6,x3],dim=1)
        
        x6 = self.up4(x6)
        x6 = torch.cat([x6,x2],dim=1)
        
        x6 = self.up5(x6)
        x6 = torch.cat([x6,x1],dim=1)
        
        
        x6 = torch.tanh(self.last(x6))
        return x6
  
# Define the discriminator  
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.down1 = Downsample(6,64)
        self.down2 = Downsample(64,128)
        self.conv1 = nn.Conv2d(128,256,3)
        self.bn = nn.BatchNorm2d(256)
        self.last = nn.Conv2d(256,1,3)
    def forward(self,anno,img):
        x=torch.cat([anno,img],axis =1) 
        x=self.down1(x,is_bn=False)
        x=self.down2(x,is_bn=True)
        x=F.dropout2d(self.bn(F.leaky_relu(self.conv1(x))))
        x=torch.sigmoid(self.last(x))  #batch*1*60*60
        return x
  
device = "cuda" if torch.cuda.is_available() else'cpu'
gen = Generator().to(device)
dis = Discriminator().to(device)
d_optimizer = torch.optim.Adam(dis.parameters(),lr=1e-3,betas=(0.5,0.999))
g_optimizer = torch.optim.Adam(gen.parameters(),lr=1e-3,betas=(0.5,0.999))

```

##### 4.3 Training the Model, Let's Begin!

I have set 20 epochs, as shown below:

```python
# Plotting
def generate_images(model,test_anno,test_real):
    prediction = model(test_anno).permute(0,2,3,1).detach().cpu().numpy()
    test_anno = test_anno.permute(0,2,3,1).cpu().numpy()
    test_real = test_real.permute(0,2,3,1).cpu().numpy()
    plt.figure(figsize = (10,10))
    display_list = [test_anno[0],test_real[0],prediction[0]]
    title = ['Input','Ground Truth','Output']
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off') # Turn off the coordinate system
    plt.show()
 
test_imgs_path = glob.glob('extended/*.jpg')
test_annos_path = glob.glob('extended/*.png')
 
test_dataset = CMP_dataset(test_imgs_path,test_annos_path)
 
test_dataloader = torch.utils.data.DataLoader(
test_dataset,
batch_size=BATCHSIZE,)
 
# Define the loss function
# CGAN loss function
loss_fn = torch.nn.BCELoss()
# L1 loss
 
num_epoch = 100
annos_batch,imgs_batch = annos_batch.to(device),imgs_batch.to(device)
LAMBDA = 7  # Weight of L1 loss
 
D_loss = []# Record the changes in discriminator loss during training
G_loss = []# Record the changes in generator loss during training
  
# Start training
for epoch in range(num_epoch):
    D_epoch_loss = 0
    G_epoch_loss = 0
    count = len(dataloader)
    for step,(annos,imgs) in enumerate(dataloader):
        imgs = imgs.to(device)
        annos = annos.to(device)
        # Define the loss calculation and optimization process for the discriminator
        d_optimizer.zero_grad()
        disc_real_output = dis(annos,imgs)# Input real paired images
        d_real_loss = loss_fn(disc_real_output,torch.ones_like(disc_real_output,
                                                             device=device))
        d_real_loss.backward()
        
        gen_output = gen(annos)
        disc_gen_output = dis(annos,gen_output.detach())
        d_fack_loss = loss_fn(disc_gen_output,torch.zeros_like(disc_gen_output,
                                                              device=device
))
        d_fack_loss.backward()
        
        disc_loss = d_real_loss+d_fack_loss# Loss calculation for discriminator
        d_optimizer.step()
        
        # Define the loss calculation and optimization process for the generator
        g_optimizer.zero_grad()
        disc_gen_out = dis(annos,gen_output)
        gen_loss_crossentropyloss = loss_fn(disc_gen_out,
                                            torch.ones_like(disc_gen_out,
                                                              device=device))
        gen_l1_loss = torch.mean(torch.abs(gen_output-imgs))
        gen_loss = gen_loss_crossentropyloss +LAMBDA*gen_l1_loss
        gen_loss.backward() # Backpropagation
        g_optimizer.step() # Optimization
        
        # Accumulate loss for each batch
        with torch.no_grad():
            D_epoch_loss +=disc_loss.item()
            G_epoch_loss +=gen_loss.item()
            
        print('Training processing...')
    
    # Calculate average loss
    with torch.no_grad():
            D_epoch_loss /=count
            G_epoch_loss /=count
            D_loss.append(D_epoch_loss)
            G_loss.append(G_epoch_loss)
            # After training an epoch, print a prompt and plot the generated images
            print("Epoch:",epoch)
            generate_images(gen,annos_batch,imgs_batch)

plt.figure()        
plt.plot(range(num_epoch),D_loss,label='D_loss',color = 'red')
plt.plot(range(num_epoch),G_loss,label='G_loss',color = 'blue')
plt.show()
```

##### 4.4 Training Results

At present, my dataset is not that extensive, and the current training results as well as the loss function curves are as follows:
