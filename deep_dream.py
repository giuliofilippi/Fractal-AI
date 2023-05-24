import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    if transform is not None:
        to_tensor_transform = transforms.ToTensor()
        image = to_tensor_transform(transform(image)).unsqueeze(0)
    return image

def save_image(image, output_path):
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image.squeeze(0))
    image.save(output_path)

def deepdream(image, model, layers, iterations, lr, octave_scale, num_octaves):
    image = image.to(device)
    model = model.to(device)
    image = image.requires_grad_(True)

    octave_images = [image]
    for _ in range(num_octaves - 1):
        size = [int(octave_scale * dim) for dim in octave_images[0].shape[-2:]]
        high_freq = nn.Upsample(size=size, mode='bilinear', align_corners=False)(octave_images[-1])
        octave_images.append(high_freq)

    detail = torch.zeros_like(octave_images[-1])
    for octave_image in octave_images[::-1]:
        for _ in range(iterations):
            outputs = model(octave_image)
            loss = torch.zeros(1, device=device)
            for layer in layers:
                target_features = outputs[0][layer]
                loss += target_features.norm()

            octave_image.retain_grad()
            model.zero_grad()
            loss.backward(retain_graph=True)
            detail = lr * octave_image.grad.data / octave_image.grad.data.norm() + detail

            octave_image = octave_image + detail
            octave_image = octave_image.clamp(-1, 1)

    return octave_image

# Load pre-trained VGG model
model = models.vgg19(pretrained=True).features
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# DeepDream parameters
layers = [18] #[4, 9, 18, 27, 36]
iterations = 10 #20
lr = 0.05
octave_scale = 1.4
num_octaves = 4 #10
resize_transform = transforms.Resize((224, 224))

# Load and preprocess the input image
input_image = load_image('inputs/cat-and-dog.jpg', transform=resize_transform)
input_image = input_image.to(device)

# Apply DeepDream
output_image = deepdream(input_image, model, layers, iterations, lr, octave_scale, num_octaves)

# Save the output image
save_image(output_image, 'outputs/cat-and-dog.jpg')
