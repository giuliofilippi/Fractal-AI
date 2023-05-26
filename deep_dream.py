import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def preprocess(image):
    resize_transform = transforms.Resize((224, 224))
    transform = transforms.ToTensor()
    return transform(resize_transform(image)).unsqueeze(0)

def deprocess(tensor):
    transform = transforms.ToPILImage()
    return transform(tensor.squeeze(0).cpu())

def deepdream(model, image, iterations, lr, layers):
    image = preprocess(image).to(device)
    image.requires_grad = True

    for i in range(iterations):
        print(i)
        model.zero_grad()
        output = model(image)

        loss = torch.zeros(1, device=device)
        for layer in layers:
            target_features = output[0][layer]
            loss += target_features.norm()

        loss.backward()
        norm_grad = torch.norm(image.grad.data)
        image.data += lr * image.grad.data / norm_grad

        image.grad.data.zero_()

    return deprocess(image)

# Load pre-trained VGG16 model
model = models.vgg16(pretrained=True)
layers = [4, 9, 18, 27, 36]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load and preprocess the input image
input_image = Image.open("inputs/cat-and-dog.jpg")
output_image = deepdream(model, input_image, iterations=40, lr=0.1, layers=layers)

# Save the output image
output_image.save("outputs/cat-and-dog.jpg")
