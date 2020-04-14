# https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt

img_gallery = "/home/ninja/PycharmProjects/MedicalColorTransfer/toy_dataset/head-no-bg"
mri_img = "/home/ninja/PycharmProjects/MedicalColorTransfer/toy_dataset/radiological/mri/src-0027.png"

img1_path = mri_img
# img2_path = os.path.join(img_gallery, "ref-0027.png")


# Load the pretrained model
model = models.vgg19(pretrained=True)
# Use the model object to select the desired layer
layer = model.classifier._modules.get('0')
print(layer)

# Set model to evaluation mode
model.eval()


scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()


def sort_tuple(tup):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    return (sorted(tup, key=lambda x: x[1], reverse=True))


def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    img = img.convert('RGB')

    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(4096)

    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.squeeze())

    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)

    # 6. Run the model on our transformed image
    model(t_img)

    # 7. Detach our copy function from the layer
    h.remove()

    # 8. Return the feature vector
    return my_embedding


cos_sims = []
img1_vec = get_vector(img1_path)

for img in os.listdir(img_gallery):
    img_path = os.path.join(img_gallery, img)

    img2_vec = get_vector(img_path)

    # Using PyTorch Cosine Similarity
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim = cos(img1_vec.unsqueeze(0),
                  img2_vec.unsqueeze(0))

    cos_sims.append( (img, cos_sim.item()) )

    cos_sims = sort_tuple(cos_sims)

top_3 = cos_sims[:3]



print(top_3)
# plotting images
fig=plt.figure(figsize=(8, 8))
columns = 2
rows = 2

mri = cv2.imread(img1_path)
mri = cv2.cvtColor(mri, cv2.COLOR_BGR2RGB)

fig.add_subplot(rows, columns, 1)
plt.imshow(mri)


for i, item in enumerate(top_3):
    img_name = item[0]
    img_path = os.path.join(img_gallery, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # cv2.imshow("{0}-choice-{1}".format(img_name, i+1), img)

    fig.add_subplot(rows, columns, i+2)

    plt.imshow(img)

plt.tight_layout(True)
plt.savefig("top3-vgg19.png")
plt.show()

# cv2.waitKey(0)
# cv2.destroyAllWindows()
