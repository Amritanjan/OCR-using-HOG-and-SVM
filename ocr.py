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



def my_fun(img):
    (ret, thresh) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    high_thresh=ret
    lowThresh = 0.01*high_thresh
    edge = cv2.Canny(img, lowThresh, high_thresh)
    
    cv2.imshow("img",img)
    cv2.imshow("thresh",thresh)
    cv2.imshow("edge",edge)


    

dir="/home/amritanjan/Downloads/archive/data/training_data/0/"
f=open("zero.csv",'w')


for file in os.listdir(dir):
    msg=""
    if(file.endswith('png')):
        path=os.path.join(dir,file)
        img=cv2.imread(path,0)
        #dq_img=deskew(img)
        h=hog(img)
        for e in h:
            msg=msg+str(e)+','
        f.write(msg)
        f.write("0")
        f.write('\n')
        cv2.imshow("img",img)
        #cv2.imshow("dq_img",dq_img)
        key=cv2.waitKey(1)
        if (key==ord('q')):
            break
        
    else:
        print("not image")

f.close()
