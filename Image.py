from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split

caltech_dir = "./train"
categories=[chr(ord("A")+i) for i in range(0,26)]
nb_classes = len(categories)

image_w = 50
image_h = 50

pixels = image_h * image_w * 3

x = []
y = []

for idx,cat in enumerate(categories):
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    image_dir = caltech_dir + "/" + cat
    files = glob.glob(image_dir+"/*.jpg")
    print(cat, "파일 길이 : ", len(files))
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        x.append(data)
        y.append(label)

        if i % 700 == 0:
            print(cat, " : ", f)

x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x,y)
xy = (x_train, x_test, y_train, y_test)
np.save("./numpy_data/multi_image_data.npy", xy)

print("ok", len(y))

