from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
img = Image.open("/Users/hanbing/PycharmProjects/pytorch-tutorial/imgs/003.jpg")
tool = transforms.ToTensor()
# print(img)
tensor_img = tool(img)
# print(tensor_img)

compose = transforms.Compose([tool, transforms.RandomCrop(200)])
composeimg = compose(img)
print(composeimg)
to_img = transforms.ToPILImage()
img2 = to_img(composeimg)

plt.imshow(img2)
plt.show()


