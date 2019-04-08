# we use the pre-trained xception model for image classification,
# and then add layers to contextualize to our specific problem.

from keras.applications.xception import Xception, decode_predictions
import cv2 as cv
import numpy as np
print("Module imports successful.")



# import images
# full_path = ""~/CS/Hydroponic-Root-Classifier/"
hairy_root_path = "dataset/hairy_root/"
non_hairy_root_path = "dataset/non_hairy_root/"

test_image = cv.imread(hairy_root_path + "1_a.jpg")
# test_image2 = cv.imread(hairy_root_path + "2_a.jpg")
# test_images = [test_image, test_image2]
# for img in test_images:
#     img = np.expand_dims(img, axis=0)
xception_preprocess_input(test_image)


# build pre-trained xception model
model = Xception(include_top=False,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=(256, 256, 3),
                 pooling=None)

# expand x to the 4-dimensional tensor (batch_size, image_width, image_height, channels)
# required by the Xception model as input
test_image = np.expand_dims(test_image, axis=0)


# use xception model to make prediction for images
predictions = model.predict(test_image, batch_size=1, verbose=0, steps=None)
print("Prediction completed.")
print(predictions.shape)
