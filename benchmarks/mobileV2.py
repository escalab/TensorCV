import sys
sys.path.append('/usr/local/lib/python3.8/site-packages')

import torch
import time
import cv2
from PIL import Image
from torchvision import transforms
from torch.cuda.amp import autocast

MODE = "GPU" # GPU CPU

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = mobilenet_v3_small(weights=weights)
model.eval()

# preprocess = weights.transforms()
# print(preprocess)

preprocess1 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225])
])

preprocess2 = transforms.Compose([
    
])

preprocess3 = transforms.Compose([
    
])

execution_time = 0

for step in range(100):
    print("Step: ", step)
    print("load ../img/input"+str(step%20+1)+".jpg")

    # Torch
    print("Preprocessing")
    input_torch = Image.open("../img/input"+str(step%20+1)+".jpg")

    input_torch = preprocess1(input_torch)
    start_time = time.time()
    # input_torch = preprocess2(input_torch)
    input_batch = preprocess2(input_torch).unsqueeze(0)
    print("* Torch: %s us" % ((time.time() - start_time) * 1000000))
    # input_batch = preprocess3(input_torch).unsqueeze(0)
    
    # for GPUs
    if MODE == "GPU":
        print("Upload to GPU")
        execution_time = 0
        if torch.cuda.is_available():
            start_time = time.time()
            input_batch = input_batch.to('cuda')
            print("* GPU load: %s us" % ((time.time() - start_time) * 1000000))
            model.to('cuda')

        print("GPU computation")
        with torch.no_grad():
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            execution_time = 0
            start_time = time.time()
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            torch.cuda.synchronize()
            print("* GPU compute: %s us" % ((time.time() - start_time) * 1000000))
            
    # for CPUs
    if MODE == "CPU":
        print("CPU computation")
        execution_time = 0
        start_time = time.time()
        output = model(input_batch)
        print("* CPU: %s us" % ((time.time() - start_time) * 1000000))
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    print("\n")

    # with open("imagenet_classes.txt", "r") as f:
    #     categories = [s.strip() for s in f.readlines()]
    # top5_prob, top5_catid = torch.topk(probabilities, 5)
    # for i in range(top5_prob.size(0)):
    #     print(categories[top5_catid[i]], top5_prob[i].item()*100//1)