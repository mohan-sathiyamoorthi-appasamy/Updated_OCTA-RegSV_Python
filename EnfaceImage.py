import cv2
import numpy as np
import glob
import csv
import sys

path = glob.glob(sys.argv[1]+"\*.tif")
read_images = []

for file in path:
    inputImage = cv2.imread(file)
    gray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    read_images.append(gray)

read_images = np.array(read_images)

path1 = glob.glob(sys.argv[2]+"\*.csv")
read_layers = []

for file1 in path1:
    rows = []
    with open(file1,'r') as file:
        csvreader = csv.reader(file)
 
        for x in csvreader:
            rows.append(x)
    read_layers.append(rows)
    
read_layers = np.int_(np.array(read_layers))

l1 = int(sys.argv[4])
l2 = int(sys.argv[5])

def enfaceImage(layer):
    EnfaceOCTangio = np.mean(layer[:,:,:],axis=1)
    maxAvgVal = np.max(EnfaceOCTangio)
    mulVal = 255 / maxAvgVal
    EnfaceOCTangioImg = mulVal * EnfaceOCTangio
    EnfaceOCTangioImg = np.squeeze(EnfaceOCTangioImg)
    EnfaceOCTangioImg = np.uint8(EnfaceOCTangioImg)
    
    motionFreeAngio = cv2.resize(EnfaceOCTangioImg, (300, 300))
    return motionFreeAngio

def FFTFilter(img):
    img2 = np.fft.fftshift(np.fft.fft2(img,s=(512,512)))

    BW = np.ones([512,512])
    BW[0:250,249:258] = 0
    BW[259:512,249:258] = 0
    
    img5 = BW * img2
    
    img6 = np.fft.ifft2(np.fft.ifftshift(img5))
    img7 = np.abs(img6[0:img.shape[0],0:img.shape[1]])
    return img7

def layerSegmentation(image,layers,i,j):

    Enface = np.zeros(image.shape)

    for noBscan in range(0,image.shape[0]):
        layerPath = layers[noBscan]
        
        Layer1 = layerPath[i]
        Layer2 = layerPath[j]
       
        OCTImage = image[noBscan]
        szImg2,szImg1 = OCTImage.shape

        for i1 in range(0,szImg1):
            for j1 in range(Layer1[i1],Layer2[i1]+1):
                Enface[noBscan,j1,i1] = OCTImage[j1,i1]
      
    img = enfaceImage(Enface)
    EnfaceImage = FFTFilter(img)

    return EnfaceImage

Enface_Image = layerSegmentation(read_images,read_layers,l1,l2)

# kernel = np.array([[0, -1, 0],
#                    [-1, 5,-1],
#                    [0, -1, 0]])
# Enface_Image = cv2.filter2D(Enface_Image, ddepth=-1, kernel=kernel)

Aline_max = np.max(Enface_Image)
div = 255 / Aline_max
Final_Image = Enface_Image * div
Final_Image = np.uint8(Final_Image)
cv2.imwrite(sys.argv[3]+"./EnfaceImage.tif",Final_Image)