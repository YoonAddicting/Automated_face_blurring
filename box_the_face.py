import cv2
import numpy as np
import time
import argparse

haarcascade_path='Automated_face_blurring/src/haarcascade_frontalface_alt.xml'

def face_pixelate(image,blocks=3):
    (h,w)=image.shape[:2]
    Xsteps=np.linspace(0,w,blocks+1,dtype='int')
    Ysteps=np.linspace(0,h,blocks+1,dtype='int')
    for i in range(1,len(Ysteps)):
        for j in range(1,len(Xsteps)):
            startX=Xsteps[j-1]
            endX=Xsteps[j]
            startY=Ysteps[i-1]
            endY=Ysteps[i]
            roi=image[startY:endY,startX:endX]
            (B,G,R)=[int(x) for x in cv2.mean(roi)[:3]]
            res=cv2.rectangle(image,(startX,startY),(endX,endY),(B,G,R),-1)
    return image

def face_mask(image,opacity=0.5):
    (h,w)=image.shape[:2]
    mask=np.zeros((h,w,3),dtype='uint8')
    res = cv2.addWeighted(image, 1-opacity, mask, opacity, 1)

    return res

def face_blurring(image,blur_type,ksize=23,blocks=3,opacity=0.5):
    img_copy=np.copy(image)
    (h, w)=img_copy.shape[:2]
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(haarcascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 2)
    print('number of faces: {0}'.format(len(faces)))

    for(x,y,w,h) in faces:
        face=img_copy[y:y+h,x:x+w]
        if(blur_type=="pixelate"):
            face=face_pixelate(face,blocks)
        if(blur_type=="masking"):
            face=face_mask(face,opacity)
            img_copy[y:y+h,x:x+w]=face
        else:
            face=cv2.blur(face,(ksize,ksize))
            img_copy[y:y+h,x:x+w]=face
    return img_copy

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to input image")
    ap.add_argument("--output", required=True, help="path to output image")
    ap.add_argument("--blur_type",type=str, default="simple",choices=["blur", "pixelate","masking"],
	help="face anonymizing method")
    ap.add_argument("--ksize", type=int, default=23, help="kernel size for the blurring method")
    ap.add_argument("--blocks", type=int, default=15, help="# of blocks for the pixelated blurring method")
    ap.add_argument("--opacity", type=float, default=0.5, help="opacity of the masking method")
    args = vars(ap.parse_args())
    img = cv2.imread(args['image'])
    start_time = time.time()
    output=face_blurring(img,args['blur_type'],args['ksize'],args['blocks'], args['opacity'])
    end_time = time.time()
    t = end_time-start_time
    print('time: {0}s'.format(t))
    cv2.imwrite(args['output'], output)
