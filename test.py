import numpy as np
import cv2
import copy as copy
import time
import imutils




#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('data/sample_2.mp4')
width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH ) )
height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT ) )
fps =  int( cap.get(cv2.CAP_PROP_FPS) )
print(width)



#BackSubs = cv2.createBackgroundSubtractorKNN(dist2Threshold = 400) #objeto modelo fondoself.
#BackSubs = cv2.createBackgroundSubtractorMOG2(history = 60,
#                                             varThreshold = 20) #objeto modelo fondoself.


BackSubs = cv2.createBackgroundSubtractorMOG2(history = 240) #objeto modelo fondoself.


# BackSubs2 = cv2.createBackgroundSubtractorMOG2()
#BackSubs = cv2.createBackgroundSubtractorMOG2()
#BackSubs.setHistory(600)
#BackSubs.setNSamples(20)
#BackSubs = cv2.createBackgroundSubtractorMOG2()
#BackSubs = cv2.createBackgroundSubtractorMOG()


BackSubs.setShadowValue(0) # blanco

minBlobSize = 2000

def analisisBlobs(imBlobs):
    '''
    * analisis y filtrado de Blobs
    '''
    newContours = []
    imBlobsFilter = copy.copy(imBlobs*0) 
    imBBoxCut = copy.copy(imBlobs*0)

    contours, hierarchy = cv2.findContours(imBlobs,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    for cnt in contours:
        areas.append( cv2.contourArea(cnt) )

    #np.max(areas)

    try:
        max = np.argmax(areas, axis=0)
        x,y,w,h = cv2.boundingRect(contours[max])
        cv2.rectangle(frame,(int(x),int(y)),(int(x+w),int(y+h)),(0,0,255),3 )
    except:
        pass

    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area > minBlobSize :
    #         newContours.append(cnt)
    #         x,y,w,h = cv2.boundingRect(cnt)

    #         #h = np.divide(h,4) 
    #         #cv2.rectangle(imBBoxCut,(x,y),(x+w,y+h),255,-1)
    #         cv2.rectangle(imBBoxCut,(int(x),int(y)),(int(x+w),int(y+h)),(255,255,0),3 )



    #cv2.drawContours(imBlobsFilter, newContours, -1, 255, cv2.FILLED)
    #contours, hierarchy = cv2.findContours(imBlobsFilter,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) 
    imBlobsFinal = imBBoxCut
    #np.logical_and(imBBoxCut, imBlobsFilter,imBlobsFinal)   

    return imBlobsFinal

a = 1
ct = 0
a = 0
detec_on = False
sum_detec = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #frame = cv2.rotateImage(frame, 90)
    
    # getRotationMatrix2D(  center, angle, scale    )
    #M = cv2.getRotationMatrix2D(((width-1)/2.0,(height-1)/2.0),-90,0.5)
    #M = cv2.getRotationMatrix2D((40,500),-70,1)
    #frame = cv2.warpAffine(frame,M,(height,width))
    frame = imutils.rotate_bound(frame, 90)

    #time.sleep(0.01)


    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    estMov = BackSubs.apply(gray)
    # estMov2 = BackSubs2.apply(gray)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))     #blob mov
    ImErode = cv2.erode(estMov,kernel,iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    ImDilate = cv2.dilate(ImErode,kernel,iterations = 1)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
    # ImDilate2 = cv2.dilate(ImErode,kernel,iterations = 2)

    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))     #blob mov
    # ImErode2 = cv2.erode(estMov,kernel,iterations = 1)
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    # ImDilate2 = cv2.dilate(ImErode,kernel,iterations = 2)

    blos = analisisBlobs(ImDilate)


    height, width = frame.shape[:2]

    line = gray*0
    x1 = 0.4*width
    y1 = 00*height
    x2 = .35*width
    y2 = 1*height
    cv2.line(line,(int(x1),int(y1)),(int(x2),int(y2)),255,7)


    detec = np.logical_and(line, ImDilate)
    sum_detec += np.sum(detec)
    #np.logical_and(line, ImDilate, line)
    


    delay = 20 
    if ct == delay:
        if sum_detec > 0:
            detec_on = True
        else:
            detec_on = False

        sum_detec = 0
        ct = 0

    ct =ct+ 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'Demo Conceptual',(10,25), font, 1,(255,255,255),2,cv2.LINE_AA)
    if sum_detec > 0 :
        cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),7)
        cv2.putText(frame,'Deteccion mano en baranda',(10,450), font, 0.8,(0,0,255),2,cv2.LINE_AA)


    if detec_on :
        cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),7)
        cv2.putText(frame,'Deteccion mano en baranda',(10,450), font, 0.8,(0,0,255),2,cv2.LINE_AA)
    else :
        cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),7)
        #cv2.putText(frame,'No deteccion baranda',(10,450), font, 1,(255,255,255),2,cv2.LINE_AA)


    # Display the resulting frame
    #cv2.imshow('grray',gray)
    cv2.imshow('bs',estMov)
    #cv2.imshow('frame2',ImErode)
    cv2.imshow('dialte',ImDilate)
    cv2.imshow('blos',frame)
    #cv2.imshow('line',line*ImDilate*255)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
