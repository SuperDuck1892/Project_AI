import numpy as np
from keras.utils import np_utils, to_categorical, load_img, img_to_array
import cv2
import os
from keras.models import Sequential, load_model
from sklearn import preprocessing
from keras.utils import load_img, img_to_array

vid = cv2.VideoCapture(0)
print("Camera connection successfully established")
i = 0
model = Sequential()
new_model = load_model('face.h5')


while(True):
    r, frame = vid.read()
    cv2.imshow('frame', frame)
    cv2.imwrite('C:\\Users\\DELL\\Desktop\\AI_CK\\Data_Test'+ str(i) + ".jpg", frame)
    test_image = load_img('C:\\Users\\DELL\\Desktop\\AI_CK\\Data_Test' + str(i) + ".jpg", target_size=(150, 150))
    test_image = img_to_array(test_image)
    test_image=test_image.astype('float32')
    test_image = np.expand_dims(test_image, axis=0)
    result = (new_model.predict(test_image).argmax())
    classes = ['Nam', 'Nha', 'Nhan','Viet']

    print('Đây là : {}'.format(classes[result]))
    os.remove('C:\\Users\\DELL\\Desktop\\AI_CK\\Data_Test' + str(i) + ".jpg")
    i = i + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()



