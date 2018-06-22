import cv2
import numpy as np
import math

#Sum from n to m terms in array###############################
def sum_nm(s,n,m):
    sum = 0
    for i in range(n,m):
        sum += s[i]
    return sum
#End#############################################################

#Sum from n to m terms in array###############################
def sum_inm(s,n,m):
    sum = 0
    for i in range(n,m):
        sum += s[i]*i
    return sum
#End#############################################################


#Otsu algorithm for 1 channel##################################################
def otsu_1cha(img,iterations):
    print('Otsu algorithm')
    mask = np.zeros((np.shape(img)[0],np.shape(img)[1]))
    mask.fill(255)
    
    hist = np.zeros(256,int)
    
    N = np.shape(img)[0] * np.shape(img)[1]

    imgcopy = np.matrix.copy(img)
    
    #Iterations
    for i in range(0,iterations):
        #Get the number of pixels in each level
        for y in range(0,np.shape(img)[0]):
            for x in range(0,np.shape(img)[1]):
                hist[int(imgcopy[y,x])] += 1

        threshold = 0
        max_bv = 0

        
        for k in range(0,256):
            w0 = sum_nm(hist,0,k) / N
            w1 = sum_nm(hist,k,256) / N
            if w0 != 0 and w1 != 0:
                u0 = sum_inm(hist,0,k) / N / w0
                u1 = sum_inm(hist,k,256) / N / w1

                bv = w0*w1*math.pow((u0-u1),2)
                if bv > max_bv:
                    max_bv = bv
                    threshold = k

        #apply mask
        for y in range(0,np.shape(img)[0]):
            for x in range(0,np.shape(img)[1]):
                if imgcopy[y,x] < threshold:
                    mask[y,x] = 0
    return mask
    
#End Otsu########################################################

#Generate final foreground#######################################
def gfore(img,mb,mg,mr):
    result = np.matrix.copy(img)
    for y in range(0,np.shape(img)[0]):
        for x in range(0,np.shape(img)[1]):
            if mb[y,x] == 0 and mg[y,x] == 0 and mr[y,x] == 0:
                result[y,x] = 0

    return result
#End#############################################################


#Get the texture#################################################
def gettexture(img,N):
    #convert to gray
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    result = np.zeros((np.shape(img)[0],np.shape(img)[1]))
    
    hs = math.floor(N/2)
    for y in range(hs,np.shape(img)[0] - hs - 1):
        for x in range(hs,np.shape(img)[1] - hs - 1):
            window = gray[y-hs:y+hs,x-hs:x+hs]
            result[y,x] = int(math.sqrt(math.pow(gray[y,x] - window.mean(),2)))

    return result
#################################################################

#Contour Extraction##############################################
def contour(img):
    print('Contour extraction')
    result = np.zeros((np.shape(img)[0],np.shape(img)[1]))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    hs = math.floor(3/2)
    for y in range(hs,np.shape(img)[0] - hs - 1):
        for x in range(hs,np.shape(img)[1] - hs - 1):
            if gray[y,x] != 0:
                window = gray[y-hs:y+hs,x-hs:x+hs]
                if (0 in window):
                    result[y,x] = (255)
    return result
#End#############################################################

img1 = cv2.imread('lake.jpg')
img2 = cv2.imread('leopard.jpg')
img3 = cv2.imread('brain.jpg')


#Lake.jpg#########################################
print('Start processing image Lake.jpg')
b,g,r = cv2.split(img1)
maskb = otsu_1cha(b,4)
cv2.imwrite('Lake_fB1.png',maskb)
maskg = otsu_1cha(g,4)
cv2.imwrite('Lake_fG1.png',maskg)
maskr = otsu_1cha(r,4)
cv2.imwrite('Lake_fR1.png',maskr)
foreground = gfore(img1,maskb,maskg,maskr)
cv2.imwrite('Lake_otsu.png',foreground)
cont = contour(foreground)
cv2.imwrite('Lake_cont.png',cont)

texture3 = gettexture(img1,3)
mask3 = otsu_1cha(texture3,4)
cv2.imwrite('Lake_3N.png',mask3)
texture5 = gettexture(img1,5)
mask5 = otsu_1cha(texture5,4)
cv2.imwrite('Lake_5N.png',mask5)
texture7 = gettexture(img1,7)
mask7 = otsu_1cha(texture7,4)
cv2.imwrite('Lake_7N.png',mask7)
foreground = gfore(img1,mask3,mask5,mask7)
cv2.imwrite('Lake_texture.png',foreground)

#Leopard.jpg########################################
print('Start processing image Leopard.jpg')
b,g,r = cv2.split(img2)
maskb = otsu_1cha(b,4)
cv2.imwrite('Leo_fB1.png',maskb)
maskg = otsu_1cha(g,4)
cv2.imwrite('Leo_fG1.png',maskg)
maskr = otsu_1cha(r,4)
cv2.imwrite('Leo_fR1.png',maskr)
foreground = gfore(img2,maskb,maskg,maskr)
cv2.imwrite('Leo_otsu.png',foreground)

texture3 = gettexture(img2,3)
mask3 = otsu_1cha(texture3,4)
cv2.imwrite('Leo_3N.png',mask3)
texture5 = gettexture(img2,5)
mask5 = otsu_1cha(texture5,4)
cv2.imwrite('Leo_5N.png',mask5)
texture7 = gettexture(img2,7)
mask7 = otsu_1cha(texture7,4)
cv2.imwrite('Leo_7N.png',mask7)
foreground = gfore(img2,mask3,mask5,mask7)
cv2.imwrite('Leo_texture.png',foreground)

cont = contour(foreground)
cv2.imwrite('Leo_cont.png',cont)
#Brain.jpp#############################################
print('Start processing image Braind.jpg')
b,g,r = cv2.split(img3)
maskb = otsu_1cha(b,4)
cv2.imwrite('Brain_fB1.png',maskb)
maskg = otsu_1cha(g,4)
cv2.imwrite('Brain_fG1.png',maskg)
maskr = otsu_1cha(r,4)
cv2.imwrite('Brain_fR1.png',maskr)
foreground = gfore(img3,maskb,maskg,maskr)
cv2.imwrite('Brain_otsu.png',foreground)
cont = contour(foreground)
cv2.imwrite('Brain_cont.png',cont)

texture3 = gettexture(img3,3)
mask3 = otsu_1cha(texture3,4)
cv2.imwrite('Brain_3N.png',mask3)
texture5 = gettexture(img3,5)
mask5 = otsu_1cha(texture5,4)
cv2.imwrite('Brain_5N.png',mask5)
texture7 = gettexture(img3,7)
mask7 = otsu_1cha(texture7,4)
cv2.imwrite('Brain_7N.png',mask7)
foreground = gfore(img3,mask3,mask5,mask7)
cv2.imwrite('Brain_texture.png',foreground)
