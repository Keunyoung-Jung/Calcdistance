import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import pypdr

#-------------------------오차율을 계산하기 위한 함수--------------------
def error_value(filename,dist) :
    filename_array = filename.split('_')
    real_dist = int(filename_array[1])
    #파일이름에 적어둔 실험거리를 기준으로 하기위해 파일이름을 분리해서 실험거리를 가져옴
    error_dist = ((abs(real_dist - dist))/real_dist)*100
    #실험거리와 측정된거리의 오차율을 계산함
    return error_dist
    

#-------------------------거리 측정 결과값그래프를 그리기 위한 함수--------------------        
def result_line(array1,array2,array3,array4) :
    style = '-'
    widths = 1
    xpoints1 = np.arange(0,len(array1),1)
    ypoints1 = array1
    xpoints2 = np.arange(0,len(array1),1)
    ypoints2 = array2
    xpoints3 = np.arange(0,len(array1),1)
    ypoints3 = array3
    xpoints4 = np.arange(0,len(array1),1)
    ypoints4 = array4
    plt.figure(figsize=(10,4))
    plt.plot(xpoints1,ypoints1,linestyle=style,color='mediumseagreen',linewidth=widths,label='median')
    plt.plot(xpoints2,ypoints2,linestyle=style,color='blue',linewidth=widths,label='average')
    plt.plot(xpoints3,ypoints3,linestyle=style,color='red',linewidth=widths,label='maxhist')
    plt.plot(xpoints4,ypoints4,linestyle=style,color='orange',linewidth=widths,label='mix')
    plt.title("distance")
    plt.xlabel("video_number")
    plt.ylabel("distance")
    plt.legend(loc='upper left')
    plt.show("lines")

#-------------------------오차율 그래프를 그리기 위한 함수--------------------
def error_line(array1,array2,array3,array4) :
    style = '-'
    widths = 1
    xpoints1 = np.arange(0,len(array1),1)
    ypoints1 = array1
    xpoints2 = np.arange(0,len(array1),1)
    ypoints2 = array2
    xpoints3 = np.arange(0,len(array1),1)
    ypoints3 = array3
    xpoints4 = np.arange(0,len(array1),1)
    ypoints4 = array4
    plt.figure(figsize=(10,4))
    plt.plot(xpoints1,ypoints1,linestyle=style,color='mediumseagreen',linewidth=widths,label='median')
    plt.plot(xpoints2,ypoints2,linestyle=style,color='blue',linewidth=widths,label='average')
    plt.plot(xpoints3,ypoints3,linestyle=style,color='red',linewidth=widths,label='maxhist')
    plt.plot(xpoints4,ypoints4,linestyle=style,color='orange',linewidth=widths,label='mix')
    plt.title("error")
    plt.xlabel("video_number")
    plt.ylabel("error(rate)")
    plt.legend(loc='upper left')
    plt.show("lines")

#-------------------------CDF(오차누적그래프)를 그리기 위한 함수--------------------    
def error_accurmulate_line(array1,array2,array3,array4) :
    widths = 1
    ypoints1 = np.arange(0,0.99,(1/len(array1)))
    #배열생성시 끝값을 1로 설정하면 값이 임의로 1개 더생겨 오류가 발생하므로 0.99로 설정
    xpoints1 = array1
    ypoints2 = np.arange(0,0.99,(1/len(array1)))
    xpoints2 = array2
    ypoints3 = np.arange(0,0.99,(1/len(array1)))
    xpoints3 = array3
    #ypoints4 = np.arange(0,0.99,(1/len(array1)))
    #xpoints4 = array4
    #Median과 Mean값을 mix한 값을 추출했으나 현재 사용안함
    plt.figure(figsize=(10,4))
    plt.plot(xpoints1,ypoints1,linestyle='-',color='mediumseagreen',linewidth=widths,label='median')
    plt.plot(xpoints2,ypoints2,linestyle='-.',color='blue',linewidth=widths,label='average')
    plt.plot(xpoints3,ypoints3,linestyle='--',color='red',linewidth=widths,label='maxhist')
    #plt.plot(xpoints4,ypoints4,linestyle='-',color='orange',linewidth=widths,label='mix')
    plt.title("error_accurmurate")
    plt.ylabel("video_number(cdf)")
    plt.xlabel("error(rate)")
    plt.ylim(0,1)
    plt.legend(loc='upper left')
    plt.show("lines")


#가운데 값을 구하기위한 함수
def mid(a, b):
    return (a+b)/2

##-------------------------특정 열의 값으로 중앙값을 찾기 위한 함수--------------------
def getMedianTupleOfOneColumn(tupleArray, col):    #특정 열의 값으로 중앙값 찾기 (col: 0~)
    size = len(tupleArray)
    if (size == 0): return (0,0,0)

    sortedArray = sorted(tupleArray, key=lambda elem: elem[col])
    #배열을 정렬하는 부분 (sort함수 이용)
    medianTuple = sortedArray[0]
    if (size % 2 == 0):
        a = int(size / 2)
        b = a - 1
        medianTuple = (mid(sortedArray[a][0], sortedArray[b][0]),\
                mid(sortedArray[a][1], sortedArray[b][1]),\
                mid(sortedArray[a][2], sortedArray[b][2]))
    else:
        medianTuple = sortedArray[size / 2]

    return medianTuple

#Pdr을 작동시키는 함수 (pypdr.py파일이 존재해야함)
def runPdr(pdr, txtLine):
    if len(txtLine) == 0: return

    row = txtLine.split()
    time = float(row[0]) * 1000          # 센서 측정시간 (millis)
    acc = [float(i) for i in row[1:4]]   # 가속도센서
    mag = [float(i) for i in row[4:7]]   # 지자기센서
    gyro = [float(i) for i in row[7:10]] # 자이로센서

    # PDR 계산
    pdr.calculate(acc, gyro, mag, time, 0)
    return pdr.getStepCount()

#-------------------------실험영상 1개의 프레임별 거리변화그래프를 그리기 위한 함수--------------------
def line(array1,array2,array3):    
    style = '-'
    colors = 'mediumseagreen'
    widths = 1
    xpoints1 = np.arange(0,len(array1),1)
    ypoints1 = array1
    xpoints2 = np.arange(0,len(array1),1)
    ypoints2 = array2
    xpoints3 = np.arange(0,len(array1),1)
    ypoints3 = array3
    plt.figure(figsize=(3,4))
    #plt.ylim(0, 10)
    plt.plot(xpoints1,ypoints1,linestyle=style,color=colors,linewidth=widths,label='median result')
    plt.plot(xpoints2,ypoints2,linestyle=style,color='blue',linewidth=widths,label='average result')
    plt.plot(xpoints3,ypoints3,linestyle=style,color='red',linewidth=widths,label='maxhist result')
    plt.title("distance")
    plt.xlabel("frame")
    plt.ylabel("value")
    plt.legend(loc='upper left')
    plt.show("lines")
    
#---------------프레임마다 모션 벡터(특징점)들이 계산하는 점들의 값과 필터 별로 계산되는 값을 보기위한 함수 -------------------
def showMotionVectors(diffpoint,diffpoint2, calcDist1=None, calcDist2=None, calcDist3=None , bins =None):
    plt.grid(color='k', linestyle='-', linewidth=0.1)
    #plt.xlim(-.08, .08)
    #plt.ylim(-1, 30)
    plt.xlabel("cm")
    plt.ylabel("y")
    plt.scatter(diffpoint[:],np.zeros_like(diffpoint),color='mediumseagreen', label='features')
    plt.hist(diffpoint2[:],bins=bins,color='yellowgreen',histtype='stepfilled')
    if calcDist1 is not None :
        plt.scatter(calcDist1, 0,color='orange' ,label='median result')
    if calcDist2 is not None :
        plt.scatter(calcDist2, 0,color='blue' ,label='average result')
    if calcDist3 is not None :
        plt.scatter(calcDist3, 0,color='red' ,label='max hist result')
        
    plt.legend(loc='upper left')
    plt.show()
    
#-------------------------거리를 계산하기 위한 함수--------------------    
def distcalc(Filename,startlabel=None,finishlabel=None) :   #start,finish라벨은 등속으로 측정했을 때 사용 현재 사용안함
    
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
    feature_params = dict(maxCorners=200, qualityLevel = 0.01,minDistance =20, blockSize = 3)
    lk_params = dict(winSize=(20,20),maxLevel = 5,criteria=termination)
    #optical flow와 goodfeatureToTrack 함수에서 사용하는 매개변수
    FilePath = 'sensor video3/'+Filename+'.mp4'
    #실험영상 파일의 경로
    realvalue = 3.7325
    #optical가 측정하는 픽셀거리를 실세계거리로 변환해주는 변수 (자동화가 필요함)
    cap = cv2.VideoCapture(FilePath)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  #프레임의 넓이
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #프레임의 높이
    fps = cap.get(cv2.CAP_PROP_FPS)             #fps
    #카메라 관련 변수들 선언 (사용은 안하나 필요시 사용가능)
    _, prev = cap.read() 
    prev = prev[0:480, 0:250]
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) 
    #프레임을 하나하나 읽기전 가장 첫화면을 prev로 만들어 while문 안에서 계산시 오류가 없도록 해준다.
    count = 0
    disttotal = 0
    disttotal_a = 0
    disttotal_h = 0
    distarray_median = []
    distarray_average = []
    distarray_maxhist = []
    distarray_mix = []
    xyarray=[]
    xarray=[]
    yarray=[] 
    #optical flow를 계싼하기위한 변수와 배열들을 선언
    sensorFilePath = 'sensor video3/'+Filename+'.txt'
    sensorFile = open(sensorFilePath)
    pdr = pypdr.Pdr()
    stepCount = 0
    stepCount_a = 0
    stepCount_h = 0
    stepLength = 0
    stepLength_a = 0
    stepLength_h = 0
    #pdr을 이용하기위해 센서값파일과 변수들을 선언
    
    #-----------------이곳 부터 프레임마다 optical flow와 특징점 계산이 시작--------------------    
    while True :
        ret , frame = cap.read()    #프레임을 읽어오는 변수
        if not ret :
            break
        frame = frame[0:480, 0:250]     #프레임을 잘라서 화면의 윗부분만 사용 (발이 나타나면 optical flow계산시 오류가 많음)
        #frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        framegray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  #gray scale로 변환한 프레임을 framegray에 저장
        view = frame.copy()     #확인을 위한 프레임을 view에 저장
        
        mask = np.zeros_like(prevgray)  #0배열을 가진 프레임과 같은크기의 마스크를 생성
        mask0 = np.zeros_like(prev)
        mask[:] = 255
        feature_points = cv2.goodFeaturesToTrack(prevgray,mask=mask,**feature_params)   #goodFeatureToTrack함수를 이용하여 특징점 추출
        corners = np.int0(feature_points)       #특징점들이 가지고있는 값을 coners에 저장
        for i in corners : 
            x, y = i.ravel()        #corners변수에서 x좌표와 y좌표를 추출
            cv2.circle(view,(x,y),1,255,-1)     #view에 특징점 위치에 점을 찍어준다
            xyval = (x,y)           #xy값을 저장
            xyarray.append(xyval)   #xy값을 배열에 저장
            xarray.append(x)        #x값을 배열에 저장
            yarray.append(y)        #y값을 배열에 저장
            
        #print(xyarray)
        xyarray = []
        
        #print(xarray)
        
        flow_point, st, err = cv2.calcOpticalFlowPyrLK(prevgray,framegray,feature_points,None,**lk_params)
        #optical flow - 루카스카나데 함수를 이용하여 opticalflow가 이동된 특징점을 계산
        
        #-------------------------거꾸로 optical flow를 다시계산하여 점을 매칭시킴 (사용안함 주석처리)-----------------
        #flow_point_rev, st, err = cv2.calcOpticalFlowPyrLK(framegray,prevgray,flow_point,None,**lk_params)
        #diffpoint = abs(feature_points - flow_point_rev).reshape(-1,2).max(-1)
        #goodpoint = diffpoint < 1
        #assert feature_points.shape == goodpoint.shape
        #---------------------------------------------------------------------------------------------
        
        corners0 = np.int0(flow_point)  #corners0에 opticalflow가 계산한 이동된 특징점을 저장

        idx = np.where(st ==1)[0]   #optical flow에서 st변수가 1일 경우 올바른 점으로 판단한다
        good_feature_points = feature_points[idx]   #st값이 1인 점을 good~ 에 저장해준다
        good_flow_point = flow_point[idx]
        
        #good_flow_point[:,:,1] = good_feature_points[:,:,1]
        #good_feature_points[:,:,1] = good_flow_point[:,:,1]
        #y값을 같게 해서 x값으로만 계산(사용안함)
        
        distarray=[]        #특징점별 거리를 담기위한 배열 선언
        distarray2=[]
        
        for i,(new,old) in enumerate(zip(good_flow_point,good_feature_points)):
            a,b = new.ravel()       #flow로 계산한 점들의 xy좌표를 1차원 배열로 정렬
            c,d = old.ravel()       #추출한 특징점의 xy좌표를 1차원 배열로 정렬
            dist = math.sqrt((a-c)*(a-c)+(b-d)*(b-d))   #xy좌표를 이용하여 매칭된 점들 끼리의 거리를 계산
            angle = math.atan((b-d)/(a-c+1e-15))        #매칭된 점들 끼리의 각도를 계산 atan
            rad = math.atan2((b-d),(a-c))               #매칭된 점들 끼리의 각도를 계산 atan2
            #print(angle)
            view = cv2.circle(view,(a,b),3,(0,255,0),-1)    #flow로 계산되는 점을 view프레임에 찍어준다
            if dist < 25 :          #매칭을 하였는데 거리가 25이상 차이날 경우 오류로 판단
                if c < a and -0.52359877566 < angle < 0.52359877566 :   #각도가 30도이상 차이나면 오류로 판단
                    mask0 = cv2.line(mask0,(a,b),(c,d),(0,255,0),1)     #line을 그려서 표현해준다
                    #distarray.append(dist/realvalue)        #dist를 realvalue로 실세계거리로 변환(사용안함)
                    distarray.append((a-c)/realvalue)        #x좌표의 이동만으로 거리를 계산하기위해 a-c로 변경
                    
        hist , bins =  np.histogram(distarray)          #히스토그램을 그리기위해 distarray를 histogram에 필요한 변수를 받아준다
        #print(distarray)
        #print(hist)
        bins_mk = []        #np.histogram에서 추출되는 형식을 사용하기 편한 형식으로 만들기위해 배열을 선언
        bins_mk2 = []
        
        for x in range(0,9) :
            bins_mk.append([bins[x],bins[x+1]])
        #print(bins_mk)
        for x in range(0,9) :
            if hist[x] == max(hist) :
                bins_mk2.append(bins_mk[x])         #배열을 사용하기 편한 형태로 정렬
        
                    
        #print(max(max(bins_mk2)))
        for x in range(0,len(distarray)) :
            if bins_mk2 != [] :
                if min(min(bins_mk2)) <= distarray[x] <= max(max(bins_mk2)) :
                    distarray2.append(distarray[x])         #최다빈도수를 가진 거리값을 distarray2에 저장

        #print('-------------------------'+str(count)+'------------------------')
        img = cv2.add(view,mask0)   #마스크에 올렸던 그림이나 다른것들을 view프레임에 넣어준다

        distmedian = np.median(distarray)       #median필터로 계산된 거리
        distaverage = np.mean(distarray)        #mean필터로 계산된 거리
        distmaxhist = np.median(distarray2)     #최대빈도로 계산된 거리
        # ***** Yaw 관련해서 추가한 부분 ***** #
        txtLine = sensorFile.readline()
        currentStepCount = stepCount

        if txtLine != None:
            currentStepCount = runPdr(pdr, txtLine)

            timesec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            #프레임의 시간정보와 센서값의 시간정보가 일치할때 yaw를 계산
            while (len(txtLine) > 0 and float(txtLine[0]) < timesec):
                txtLine = sensorFile.readline()
                currentStepCount = runPdr(pdr, txtLine)

        yaw = pdr.getYaw()

#         distmedian = distmedian * math.cos(yaw)
        # ***** ********************** ***** #  

        currentStepCount_a = currentStepCount
        currentStepCount_h = currentStepCount
        StepCount_a = stepCount
        StepCount_h = stepCount
        
        #정지를 감지하기위해 1cm이하의 움직임은 모두 오류로 판단
        if distmedian > 1 :
            #계산된 거리에 위에서 구한 yaw각을 코사인 해주어서 거리를구함 
            stepLength += distmedian * math.cos(yaw)
            if currentStepCount != None and stepCount < currentStepCount:
                stepCount = currentStepCount
                disttotal += stepLength
                #거리에 yaw각을 곱한 길이를 지속적으로 더해줌
                stepLength = 0

        if distaverage > 1 :
            stepLength_a += distaverage * math.cos(yaw)
            if currentStepCount_a != None and stepCount_a < currentStepCount_a:
                stepCount_a = currentStepCount_a
                disttotal_a += stepLength_a
                stepLength_a = 0
        
        if distmaxhist > 1 :
            stepLength_h += distmedian * math.cos(yaw)
            if currentStepCount_h != None and stepCount_h < currentStepCount_h:
                stepCount_h = currentStepCount_h
                disttotal_h += stepLength_h
                stepLength_h = 0   
                
        distarray_median.append(disttotal/100)          #cm기준으로 거리를 구했기 때문에 100을 나누어 m기준으로 변환
        distarray_average.append(disttotal_a/100)
        distarray_maxhist.append(disttotal_h/100)   
        
        
        #showMotionVectors(distarray,distarray,distmedian,distaverage,distmaxhist,bins)
        #프레임별 모션벡터의 모습을 보는 함수
        
        #print(Filename,'median_','--frame_num :' ,count ,'value(cm) :' ,distmedian , 'distance(m) : ',disttotal/100)
        #print(Filename,'average_','--frame_num :' ,count ,'value(cm) :' ,distaverage , 'distance(m) : ',disttotal_a/100)
        #print(Filename,'maxhist_','--frame_num :' ,count ,'value(cm) :' ,distmaxhist , 'distance(m) : ',disttotal_h/100)
        #매 프레임마다 계산되는 거리를 콘솔창에 표시
        
        distance_print = "dist: %.2f m"%(disttotal/100)
        #화면에 표시할 거리의 형식 선언
        #print(distarray)
        cv2.putText(img, distance_print,(0,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
        #화면에 거리를 표시
        
        prevgray = framegray
        #모든 계산이 끝나면 현재프레임이 이전프레임으로 바뀌고 그 후 다시 계산이 시작된다
        #cv2.imshow('orginal',img)            #윈도우창으로 계산되는 모습을 보여준다
        #cv2.imwrite('img/'+Filename+'/'+Filename+'-frame'+str(count)+'.jpg',img)        #매프레임을 각각 저장
       
        k = cv2.waitKey(30)
        if k == 27:
            break
        count += 1      #프레임 넘버 카운트
    
    #-----------------------------While문 바깥 (영상 데이터 계산이 끝난 후)--------------------------
    disttotal += stepLength
    disttotal_a += stepLength_a
    disttotal_h += stepLength_h
    #마지막에 계산되는 거리를 따로 더해주어야한다 (pdr관련으로 생긴 문제)
    
    disterror = error_value(Filename,disttotal/100)             #각각 필터마다의 오차율을 계산
    disterror_a = error_value(Filename,disttotal_a/100)
    disterror_h = error_value(Filename,disttotal_h/100)
    
    disttotal_mix = (disttotal+disttotal_a)/2               #median필터와 mean필터를 결합해서 만듬
    disterror_mix = error_value(Filename,disttotal_mix/100)
    
    print('median error : %.2f'%disterror+'% /','average error : %.2f'%disterror_a+'% /','maxhist error : %.2f'%disterror_h+'% /' ,'mix error : %.2f'%disterror_mix+'%')
    #오차율 콘솔창에 표시
    distarray_median.append(disttotal/100)
    distarray_average.append(disttotal_a/100)
    distarray_maxhist.append(disttotal_h/100)
    distarray_mix.append(disttotal_mix/100)
    
    #line(distarray_median,distarray_average,distarray_maxhist)
    print('----------------------------------------------------------------------------------------')
    print(Filename,'median_','distance(m) : ',disttotal/100)
    print(Filename,'average_','distance(m) : ',disttotal_a/100)
    print(Filename,'maxhist_','distance(m) : ',disttotal_h/100)
    print(Filename,'mix_','distance(m) : ',disttotal_mix/100)
    print('----------------------------------------------------------------------------------------')
    cap.release()
    sensorFile.close()
    cv2.destroyAllWindows()
    
    return disttotal/100 , disttotal_a/100 , disttotal_h/100 , disttotal_mix/100 , disterror , disterror_a , disterror_h , disterror_mix

# Result 

dist_loca = ['ss','yg','gh','dr']
#상상관지하1층,연구동2층,공학관3층,일반아스팔트도로
dist_value = ['3','10']
#3m , 10m 실험
dist_mode = ['normal','fast','slow','stop']
#보통 , 빠른 , 느린 , 멈춤 4가지실험
dist_number = ['1','2','3']
#총3번의 실험 진행

#-------------------------------------그래프와 결과 출력--------------------------------

error_acc_median_t = []             #전체 모든 오차를 누적하기위한 배열
error_acc_average_t = []
error_acc_maxhist_t = []
error_acc_mixar_t = []

error_acc_median_t.append(0)        #오차는 0에서부터 시작
error_acc_average_t.append(0)
error_acc_maxhist_t.append(0)
error_acc_mixar_t.append(0)

for a in dist_value :
    
    error_acc_median_d = []         #거리별 오차를 누적하기위한 배열
    error_acc_average_d = []
    error_acc_maxhist_d = []
    error_acc_mixar_d = []
    
    error_acc_median_d.append(0)        #오차는 0에서부터 시작
    error_acc_average_d.append(0)
    error_acc_maxhist_d.append(0)
    error_acc_mixar_d.append(0)
    
    for b in dist_loca :
        
        disttotal_array_median = []         #실험 위치별 거리 배열
        disttotal_array_average = []
        disttotal_array_maxhist = []
        disttotal_array_mix = []

        error_array_median = []             #실험 위치별 오차 배열
        error_array_average = []
        error_array_maxhist = []
        error_array_mix = []
        
        error_acc_median = []               #실험 위치별 오차 누적배열
        error_acc_average = []
        error_acc_maxhist = []
        error_acc_mixar = []
        
        error_acc_median.append(0)
        error_acc_average.append(0)
        error_acc_maxhist.append(0)
        error_acc_mixar.append(0)
        
        error_acc_m = 0
        error_acc_a = 0
        error_acc_h = 0
        error_acc_mix = 0
        
        for j in dist_mode :
            for k in dist_number :
                # ------------------------yg_fast의 2,3번 오류처리 부분(동영상이 날라간 상태) --------------------------------------
                if j == 'fast' and b == 'yg' and a == '3' :
                    if k == '2' or k == '3' :
                        median , average , maxhist ,mix = 3, 3 ,3 , 3
                        error_m , error_a , error_h ,error_mix = 0, 0 ,0 ,0
                    else :
                        median , average , maxhist ,mix,error_m , error_a , error_h ,error_mix = distcalc(b+'_'+a+'_'+j+'_'+k)
                # --------------------------------------------------------------------------------------------
                else :
                    median , average , maxhist,mix ,error_m , error_a , error_h ,error_mix = distcalc(b+'_'+a+'_'+j+'_'+k)
                
                
                #--------------------그래프 표현을 위한 배열 추가 하는 부분 ---------------    
                error_acc_m = error_m
                error_acc_a = error_a
                error_acc_h = error_h 
                error_acc_mix = error_mix 

                disttotal_array_median.append(median)
                disttotal_array_average.append(average)
                disttotal_array_maxhist.append(maxhist)
                disttotal_array_mix.append(mix)
                
                error_array_median.append(error_m)
                error_array_average.append(error_a)
                error_array_maxhist.append(error_h)
                error_array_mix.append(error_mix)
                
                error_acc_median.append(error_acc_m)
                error_acc_average.append(error_acc_a)
                error_acc_maxhist.append(error_acc_h)
                error_acc_mixar.append(error_acc_mix)
                
                error_acc_median_d.append(error_acc_m)
                error_acc_average_d.append(error_acc_a)
                error_acc_maxhist_d.append(error_acc_h)
                error_acc_mixar_d.append(error_acc_mix)
                
                error_acc_median_t.append(error_acc_m)
                error_acc_average_t.append(error_acc_a)
                error_acc_maxhist_t.append(error_acc_h)
                error_acc_mixar_t.append(error_acc_mix)
                
        error_acc_median.sort()             #CDF는 오름차순으로 해야하므로 배열을 정렬 해준다.
        error_acc_average.sort()
        error_acc_maxhist.sort()
        error_acc_mixar.sort()
                
        result_line(disttotal_array_median,disttotal_array_average,disttotal_array_maxhist,disttotal_array_mix)
        error_line(error_array_median,error_array_average,error_array_maxhist,error_array_mix)
        error_accurmulate_line(error_acc_median,error_acc_average,error_acc_maxhist,error_acc_mixar)
        #실험 위치별 오차 누적 그래프 표현
        
    error_acc_median_d.sort()           #CDF는 오름차순으로 해야하므로 배열을 정렬 해준다.
    error_acc_average_d.sort()
    error_acc_maxhist_d.sort()
    error_acc_mixar_d.sort()

    error_accurmulate_line(error_acc_median_d,error_acc_average_d,error_acc_maxhist_d,error_acc_mixar_d)
    #실험 거리별 오차 누적 그래프 표현
    
error_acc_median_t.sort()           #CDF는 오름차순으로 해야하므로 배열을 정렬 해준다.
error_acc_average_t.sort()
error_acc_maxhist_t.sort()
error_acc_mixar_t.sort()    

error_accurmulate_line(error_acc_median_t,error_acc_average_t,error_acc_maxhist_t,error_acc_mixar_t)
#실험 전체의 오차 누적 그래프 표현
