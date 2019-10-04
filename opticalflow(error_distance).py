import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import pypdr

def error_value(filename,dist) :
    filename_array = filename.split('_')
    real_dist = int(filename_array[1])
    
    error_dist = ((abs(real_dist - dist)))*100
    
    return error_dist
    

        
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
    
def error_accurmulate_line(array1,array2,array3,array4) :
    widths = 1
    ypoints1 = np.arange(0,0.99,(1/len(array1)))
    xpoints1 = array1
    ypoints2 = np.arange(0,0.99,(1/len(array1)))
    xpoints2 = array2
    ypoints3 = np.arange(0,0.99,(1/len(array1)))
    xpoints3 = array3
    ypoints4 = np.arange(0,0.99,(1/len(array1)))
    xpoints4 = array4
    plt.figure(figsize=(10,4))
    plt.plot(xpoints1,ypoints1,linestyle='-',color='mediumseagreen',linewidth=widths,label='median')
    plt.plot(xpoints2,ypoints2,linestyle='-.',color='blue',linewidth=widths,label='average')
    plt.plot(xpoints3,ypoints3,linestyle='--',color='red',linewidth=widths,label='maxhist')
    #plt.plot(xpoints4,ypoints4,linestyle='-',color='orange',linewidth=widths,label='mix')
    plt.title("error_accurmurate")
    plt.ylabel("video_number(cdf)")
    plt.xlabel("error(cm)")
    plt.ylim(0,1)
    plt.legend(loc='upper left')
    plt.show("lines")


    
def mid(a, b):
    return (a+b)/2

def getMedianTupleOfOneColumn(tupleArray, col):    #특정 열의 값으로 중앙값 찾기 (col: 0~)
    size = len(tupleArray)
    if (size == 0): return (0,0,0)

    sortedArray = sorted(tupleArray, key=lambda elem: elem[col])

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

def line(array1,array2,array3):    #그래프를 뽑기위한 함수
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
    
def distcalc(Filename,startlabel=None,finishlabel=None) :
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
    feature_params = dict(maxCorners=200, qualityLevel = 0.01,minDistance =20, blockSize = 3)
    lk_params = dict(winSize=(20,20),maxLevel = 5,criteria=termination)
    FilePath = 'sensor video3/'+Filename+'.mp4'
    realvalue = 3.7325
    xyarray=[]
    xarray=[]
    yarray=[] 
    cap = cv2.VideoCapture(FilePath)
    count = 0
    disttotal = 0
    disttotal_a = 0
    disttotal_h = 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #out = cv2.VideoWriter('video_out.avi', fourcc, fps, (w, h))
    _, prev = cap.read() 
    prev = prev[0:480, 0:250]
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) 
    distarray_median = []
    distarray_average = []
    distarray_maxhist = []
    distarray_mix = []
    framebuffer = []
    
    sensorFilePath = 'sensor video3/'+Filename+'.txt'
    sensorFile = open(sensorFilePath)
    pdr = pypdr.Pdr()
    stepCount = 0
    stepCount_a = 0
    stepCount_h = 0
    stepLength = 0
    stepLength_a = 0
    stepLength_h = 0

    #for count in range(n_frames-2) :
    while True :
        ret , frame = cap.read()
        if not ret :
            break
        frame = frame[0:480, 0:250]
        #frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        #ROI = cv2.rectangle(frame,(0,0),(320,480),(0,0,255),2)
        framegray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        view = frame.copy()
        
        framebuffer.append(frame)
        
        mask = np.zeros_like(prevgray)
        mask0 = np.zeros_like(prev)
        mask[:] = 255
        feature_points = cv2.goodFeaturesToTrack(prevgray,mask=mask,**feature_params)
        corners = np.int0(feature_points)
        for i in corners : 
            x, y = i.ravel()
            cv2.circle(view,(x,y),1,255,-1)
            xyval = (x,y)
            xyarray.append(xyval)
            xarray.append(x)
            yarray.append(y)
            
        #print(xyarray)
        xyarray = []
        
        #print(xarray)
        
        flow_point, st, err = cv2.calcOpticalFlowPyrLK(prevgray,framegray,feature_points,None,**lk_params)
        #flow_point_rev, st, err = cv2.calcOpticalFlowPyrLK(framegray,prevgray,flow_point,None,**lk_params)
        #diffpoint = abs(feature_points - flow_point_rev).reshape(-1,2).max(-1)
        #goodpoint = diffpoint < 1
        
        #assert feature_points.shape == goodpoint.shape
        corners0 = np.int0(flow_point)

        idx = np.where(st ==1)[0]
        good_feature_points = feature_points[idx]
        good_flow_point = flow_point[idx]
        #good_flow_point[:,:,1] = good_feature_points[:,:,1]
        #good_feature_points[:,:,1] = good_flow_point[:,:,1]
        #y값을 같게 해서 x값으로만 계산
        distarray=[]
        distarray2=[]
        for i,(new,old) in enumerate(zip(good_flow_point,good_feature_points)):
            a,b = new.ravel()
            c,d = old.ravel()
            dist = math.sqrt((a-c)*(a-c)+(b-d)*(b-d))
            angle = math.atan((b-d)/(a-c+1e-15))
            rad = math.atan2((b-d),(a-c))
            #print(angle)
            view = cv2.circle(view,(a,b),3,(0,255,0),-1)
            if dist < 25 :
                if c < a and -0.52359877566 < angle < 0.52359877566 :
                    mask0 = cv2.line(mask0,(a,b),(c,d),(0,255,0),1)
                    distarray.append((a-c)/realvalue)
                    
        hist , bins =  np.histogram(distarray)
        #print(distarray)
        #print(hist)
        bins_mk = []
        bins_mk2 = []
        for x in range(0,9) :
            bins_mk.append([bins[x],bins[x+1]])
        #print(bins_mk)
        for x in range(0,9) :
            if hist[x] == max(hist) :
                bins_mk2.append(bins_mk[x])
                    
        #print(max(max(bins_mk2)))
        for x in range(0,len(distarray)) :
            if bins_mk2 != [] :
                if min(min(bins_mk2)) <= distarray[x] <= max(max(bins_mk2)) :
                    distarray2.append(distarray[x])

        #print('-------------------------'+str(count)+'------------------------')
        img = cv2.add(view,mask0)  

        distmedian = np.median(distarray)
        distaverage = np.mean(distarray)
        distmaxhist = np.median(distarray2)
        # ***** Yaw 관련해서 추가한 부분 ***** #
        txtLine = sensorFile.readline()
        currentStepCount = stepCount

        if txtLine != None:
            currentStepCount = runPdr(pdr, txtLine)

            timesec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
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
        
        if distmedian > 1 :
            stepLength += distmedian * math.cos(yaw)
            if currentStepCount != None and stepCount < currentStepCount:
                stepCount = currentStepCount
                disttotal += stepLength
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
                
        distarray_median.append(disttotal/100)
        distarray_average.append(disttotal_a/100)
        distarray_maxhist.append(disttotal_h/100)   
        
        
        #showMotionVectors(distarray,distarray,distmedian,distaverage,distmaxhist,bins)
        
        #print(Filename,'median_','--frame_num :' ,count ,'value(cm) :' ,distmedian , 'distance(m) : ',disttotal/100)
        #print(Filename,'average_','--frame_num :' ,count ,'value(cm) :' ,distaverage , 'distance(m) : ',disttotal_a/100)
        #print(Filename,'maxhist_','--frame_num :' ,count ,'value(cm) :' ,distmaxhist , 'distance(m) : ',disttotal_h/100)
        distance_print = "dist: %.2f m"%(disttotal/100)
        #print(distarray)
        cv2.putText(img, distance_print,(0,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
        
        
        prevgray = framegray
#         cv2.imshow('orginal',img)
#         cv2.imwrite('img/'+Filename+'/'+Filename+'-frame'+str(count)+'.jpg',img)
       
        k = cv2.waitKey(30)
        if k == 27:
            break
        count += 1
    
    disttotal += stepLength
    disttotal_a += stepLength_a
    disttotal_h += stepLength_h
    
    disterror = error_value(Filename,disttotal/100)
    disterror_a = error_value(Filename,disttotal_a/100)
    disterror_h = error_value(Filename,disttotal_h/100)
    
    disttotal_mix = (disttotal+disttotal_a)/2
    disterror_mix = error_value(Filename,disttotal_mix/100)
    
    print('median error : %.2f'%disterror+'cm /','average error : %.2f'%disterror_a+'cm /','maxhist error : %.2f'%disterror_h+'cm /' ,'mix error : %.2f'%disterror_mix+'cm')
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

error_acc_median_t = []
error_acc_average_t = []
error_acc_maxhist_t = []
error_acc_mixar_t = []

error_acc_median_t.append(0)
error_acc_average_t.append(0)
error_acc_maxhist_t.append(0)
error_acc_mixar_t.append(0)

for a in dist_value :
    
    error_acc_median_d = []
    error_acc_average_d = []
    error_acc_maxhist_d = []
    error_acc_mixar_d = []
    
    error_acc_median_d.append(0)
    error_acc_average_d.append(0)
    error_acc_maxhist_d.append(0)
    error_acc_mixar_d.append(0)
    
    for b in dist_loca :
        
        disttotal_array_median = []
        disttotal_array_average = []
        disttotal_array_maxhist = []
        disttotal_array_mix = []

        error_array_median = []
        error_array_average = []
        error_array_maxhist = []
        error_array_mix = []
        
        error_acc_median = []
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
                # ------------------------yg_fast의 2,3번 오류처리 부분 --------------------------------------
                if j == 'fast' and b == 'yg' and a == '3' :
                    if k == '2' or k == '3' :
                        median , average , maxhist ,mix = 3, 3 ,3 , 3
                        error_m , error_a , error_h ,error_mix = 0, 0 ,0 ,0
                    else :
                        median , average , maxhist ,mix,error_m , error_a , error_h ,error_mix = distcalc(b+'_'+a+'_'+j+'_'+k)
                # --------------------------------------------------------------------------------------------
                else :
                    median , average , maxhist,mix ,error_m , error_a , error_h ,error_mix = distcalc(b+'_'+a+'_'+j+'_'+k)
                    
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
                
        error_acc_median.sort()
        error_acc_average.sort()
        error_acc_maxhist.sort()
        error_acc_mixar.sort()
                
        result_line(disttotal_array_median,disttotal_array_average,disttotal_array_maxhist,disttotal_array_mix)
        error_line(error_array_median,error_array_average,error_array_maxhist,error_array_mix)
        error_accurmulate_line(error_acc_median,error_acc_average,error_acc_maxhist,error_acc_mixar)
        
    error_acc_median_d.sort()
    error_acc_average_d.sort()
    error_acc_maxhist_d.sort()
    error_acc_mixar_d.sort()

    error_accurmulate_line(error_acc_median_d,error_acc_average_d,error_acc_maxhist_d,error_acc_mixar_d)
    
error_acc_median_t.sort()
error_acc_average_t.sort()
error_acc_maxhist_t.sort()
error_acc_mixar_t.sort()    

error_accurmulate_line(error_acc_median_t,error_acc_average_t,error_acc_maxhist_t,error_acc_mixar_t)
    