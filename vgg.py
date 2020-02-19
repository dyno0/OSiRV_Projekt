from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

model = VGG16()
pic = input("Ime slike:")
path = "images\\" + pic
slika = load_img(path, target_size=(224, 224))
slika = img_to_array(slika)
slika = slika.reshape((1, slika.shape[0], slika.shape[1], slika.shape[2]))
slika = preprocess_input(slika)
objekt = model.predict(slika)
labels = decode_predictions(objekt)
labels = labels[0][0]
print('%s (%.2f%%)' % (labels[1], labels[2]*100))