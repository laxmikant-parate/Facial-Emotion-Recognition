import cv2

def detectFace(net, frame, confidence_threshold = 0.7):
    frameOpencvDNN=frame.copy()
    frameHeight=frameOpencvDNN.shape[0]
    frameWidth=frameOpencvDNN.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDNN,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False)
    net.setInput(blob)
    detections=net.forward()

    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>confidence_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDNN,(x1,y1),(x2,y2),(0,255,0),int(round(frameHeight/150)),8)
            
    return frameOpencvDNN,faceBoxes
    
faceProto='AI_SCE/opencv_face_detector.pbtxt'
faceModel='AI_SCE/opencv_face_detector_uint8.pb'
ageProto='AI_SCE/age_deploy.prototxt'
ageModel='AI_SCE/age_net.caffemodel'
genderProto='AI_SCE/gender_deploy.prototxt'
genderModel='AI_SCE/gender_net.caffemodel'

genderList=['Male','Female']
ageList=['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

padding=20

def getAgeGender(hasFrame, frame):
    if not hasFrame:
        cv2.waitKey()
        
    _,faceBoxes = detectFace(faceNet,frame)
    
    if not faceBoxes:
        print("No face detected")
    
    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face,1.0,(227,227),[124.96,115.97,106.13],swapRB=True,crop=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        
        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        
        cv2.putText(frame,f'{gender},{age}',(faceBox[0],faceBox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2,cv2.LINE_AA)
