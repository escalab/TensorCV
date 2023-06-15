import sys
sys.path.append('/usr/local/lib/python3.8/site-packages')

import torch
import time
import cv2
from PIL import Image
from torchvision import transforms
from torch.cuda.amp import autocast

if len(sys.argv) != 3:
    print("Usage: python3 benchmarks.py [GPU/CPU] [MODEL]")
    print("Models: Alex, Dense, Efficient, Mobile, Reg, VGG, ViT, WideRes")
    exit()

MODE = str(sys.argv[1])
MODEL = str(sys.argv[2])

if MODE != "GPU" and MODE != "CPU":
    print("Usage: python3 benchmarks.py [GPU/CPU] [MODEL]")
    print("Models: Alex, Dense, Efficient, Mobile, Reg, VGG, ViT, WideRes")
    exit()

if MODEL != "Alex" and MODEL != "Dense" and MODEL != "Efficient" and MODEL != "Mobile" and MODEL != "Reg" and MODEL != "VGG" and MODEL != "ViT" and MODEL != "WideRes":
    print("Usage: python3 benchmarks.py [GPU/CPU] [MODEL]")
    print("Models: Alex, Dense, Efficient, Mobile, Reg, VGG, ViT, WideRes")
    exit()

if MODEL == "Alex":
    from torchvision.models import alexnet,AlexNet_Weights
    weights=AlexNet_Weights.IMAGENET1K_V1
    model = alexnet(weights=weights)

if MODEL == "Dense":
    from torchvision.models import densenet121,DenseNet121_Weights
    weights=DenseNet121_Weights.IMAGENET1K_V1
    model = densenet121(weights=weights)

if MODEL == "Efficient":
    from torchvision.models import efficientnet_v2_s,EfficientNet_V2_S_Weights
    weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights)

if MODEL == "Mobile":
    from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
    weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)

if MODEL == "Reg":
    from torchvision.models import regnet_x_8gf,RegNet_X_8GF_Weights
    weights=RegNet_X_8GF_Weights.IMAGENET1K_V1
    model = regnet_x_8gf(weights=weights)

if MODEL == "VGG":
    from torchvision.models import vgg11_bn,VGG11_BN_Weights
    weights=VGG11_BN_Weights.IMAGENET1K_V1
    model = vgg11_bn(weights=weights)

if MODEL == "ViT":
    from torchvision.models import vit_l_32,ViT_L_32_Weights
    weights=ViT_L_32_Weights.IMAGENET1K_V1
    model = vit_l_32(weights=weights)

if MODEL == "WideRes":
    from torchvision.models import wide_resnet50_2,Wide_ResNet50_2_Weights
    weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1
    model = wide_resnet50_2(weights=weights)

model.eval()

# preprocess = weights.transforms()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225])
])

print(preprocess)

execution_time = 0
for step in range(100):
    print("Step: ", step)
    print("load img/input"+str(step%20+1)+".jpg")

    # Torch
    print("Preprocessing")
    input_torch = Image.open("img/input"+str(step%20+1)+".jpg")
    start_time = time.time()
    input_batch = preprocess(input_torch).unsqueeze(0)
    print("* Torch: %s ns" % ((time.time() - start_time) * 1000000000))
    
    # for GPUs
    if MODE == "GPU":
        print("Upload to GPU")
        execution_time = 0
        if torch.cuda.is_available():
            start_time = time.time()
            input_batch = input_batch.to('cuda')
            print("* GPU load: %s ns" % ((time.time() - start_time) * 1000000000))
            model.to('cuda')

        print("GPU computation")
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                execution_time = 0
                start_time = time.time()
                output = model(input_batch)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                torch.cuda.synchronize()
                print("* GPU compute: %s ns" % ((time.time() - start_time) * 1000000000))
            
    # for CPUs
    if MODE == "CPU":
        print("CPU computation")
        execution_time = 0
        start_time = time.time()
        output = model(input_batch)
        print("* CPU: %s ns" % ((time.time() - start_time) * 100000000))
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    print("\n")

    # with open("imagenet_classes.txt", "r") as f:
    #     categories = [s.strip() for s in f.readlines()]
    # top5_prob, top5_catid = torch.topk(probabilities, 5)
    # for i in range(top5_prob.size(0)):
    #     print(categories[top5_catid[i]], top5_prob[i].item()*100//1)