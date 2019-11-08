from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

'''
This class is somewhat based off of the section for generating additional data given a small dataset at 
https://medium.com/@ksusorokina/image-classification-with-convolutional-neural-networks-496815db12a8
'''

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
img = load_img('train/elephants/adventure-1822636_640.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (300, 300, 3)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 300, 300, 3)
x.shape

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='el', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
