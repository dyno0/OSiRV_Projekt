import keras
import numpy as np
from keras.applications import inception_v3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np

model = inception_v3.InceptionV3()
pic = input("Ime slike:")
path = "images\\" + pic
slika = load_img(path, target_size=(299, 299))
npSlika = img_to_array(slika)
npOrgSlika = np.expand_dims(npSlika, axis=0)
ppSlika = inception_v3.preprocess_input(npOrgSlika.copy())
outSlika = model.predict(ppSlika)
labels = decode_predictions(outSlika)
labels = labels[0][0]
print('%s (%.2f%%)' % (labels[1], labels[2]*100))