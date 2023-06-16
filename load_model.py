from joblib import load
import cv2
import numpy as np
import os
import cv2 as cv

SZ=20
bin_n = 16

def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist


# Load the saved model from file
loaded_model = load('svm_model.joblib')


dir="/home/amritanjan/Downloads/archive/data/training_data/1/"



for file in os.listdir(dir):
    msg=""
    if(file.endswith('png')):
        path=os.path.join(dir,file)
        img=cv2.imread(path,0)
        h=hog(img)
        hl=h.reshape(1, -1)
        
        prediction = loaded_model.predict(hl)
        print('Predicted class label:', prediction[0])
        cv2.imshow("img",img)
    
        key=cv2.waitKey(1)
        if (key==ord('q')):
            break
        
    else:
        print("not image")


