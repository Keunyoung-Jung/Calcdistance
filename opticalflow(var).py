import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def E_x(xarray,yarray,cnt):
    tmp = 0
    n = 0
    for i in range(0,cnt) :
        n += yarray[i]
    for i in range(0,cnt) :
        tmp += xarray[i]*yarray[i]
    E_x = tmp/n
    return E_x
    
def Variance(xarray,yarray,cnt):
    tmp = 0
    n = 0
    for i in range(0,cnt) :
        n += yarray[i]
    for i in range(0,cnt) :
        tmp += xarray[i]*xarray[i]*yarray[i]
    tmp = tmp/n
    variance = tmp - (E_x(xarray,yarray,cnt)*E_x(xarray,yarray,cnt))
    return variance

def line(array):    #그래프를 뽑기위한 함수
    style = '-'
    colors = 'mediumseagreen'
    widths = 1
    xpoints = np.arange(0,len(array),1)
    ypoints = array
    plt.figure(figsize=(3,4))
    plt.ylim(0, 10)
    plt.plot(xpoints,ypoints,linestyle=style,color=colors,linewidth=widths)
    plt.title("distance")
    plt.xlabel("frame")
    plt.ylabel("value")
    plt.show("lines")
    
def showMotionVectors(diffpoint,diffpoint2, calcDist=None, bins =None, realDist=None, prevRealDist=None):
    plt.grid(color='k', linestyle='-', linewidth=0.1)
    plt.xlim(0, 5)
    plt.ylim(-1, 60)
    plt.xlabel("cm")
    plt.ylabel("hist")
    plt.scatter(diffpoint[:],np.zeros_like(diffpoint), label='features')
    plt.hist(diffpoint2[:],bins=bins,color='yellowgreen',histtype='stepfilled')
    if calcDist is not None :
        plt.scatter(calcDist, 0,color='orange' ,label='optical flow result')
    if realDist is not None and prevRealDist is not None :
        plt.scatter(realDist - prevRealDist, 0, label='estimate')
    plt.legend(loc='upper left')
    plt.show()
    
def distcalc(Filename,startlabel=None,finishlabel=None) :
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
    feature_params = dict(maxCorners=200, qualityLevel = 0.01,minDistance =20, blockSize = 3)
    lk_params = dict(winSize=(20,20),maxLevel = 5,criteria=termination)
    FilePath = 'sensor video2/'+Filename+'.mp4'
    realvalue = 3.87
    xyarray=[]
    xarray=[]
    yarray=[] 
    cap = cv2.VideoCapture(FilePath)
    count = 0
    disttotal = 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #out = cv2.VideoWriter('video_out.avi', fourcc, fps, (w, h))
    _, prev = cap.read() 
    prev = prev[0:480, 0:250]
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) 
    distarray_median = []
    framebuffer = []
    prevdistarray = []
    
    #for count in range(n_frames-2) :
    while True :
        ret , frame = cap.read()
        if not ret :
            break
        sec = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
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
        distarray2 = []
        for i,(new,old) in enumerate(zip(good_flow_point,good_feature_points)):
            a,b = new.ravel()
            c,d = old.ravel()
            dist = math.sqrt((a-c)*(a-c)+(b-d)*(b-d))
            angle = math.atan((b-d)/(a-c))
            #print(angle)
            view = cv2.circle(view,(a,b),3,(0,255,0),-1)
            if dist < 30 :
                if c < a and -0.52359877566 < angle < 0.52359877566 :
                    mask0 = cv2.line(mask0,(a,b),(c,d),(0,255,0),1)
                    distarray.append((a-c)/realvalue)
        
        hist , bins =  np.histogram(distarray,bins=20)
        #print(distarray)
        bins_x=[]
        bins_mk = []
        bins_mk2 = []
        histarray = []
        for x in range(1,21) :
            bins_x.append(bins[x])
        
        #var=Variance(bins_x,hist,len(hist))
        #average = E_x(bins_x,hist,len(hist))
        var = np.var(distarray)
        average = np.mean(distarray)
        
        for x in range(0,20) :
            bins_mk.append([bins[x],bins[x+1]])
        #print(bins_mk)
        for x in range(0,20) :
            if hist[x] > 5 :
                if var < 0.2 :
                    if hist[x] > 10 :
                        histarray.append(hist[x])
                        bins_mk2.append(bins_mk[x])
                elif var < 0.4 :
                    if hist[x] > 15 :
                        histarray.append(hist[x])
                        bins_mk2.append(bins_mk[x])
                elif var < 0.6 :
                    if hist[x] > 20 :
                        histarray.append(hist[x])
                        bins_mk2.append(bins_mk[x]) 
                elif var < 0.8 :
                    if hist[x] > 25 :
                        histarray.append(hist[x])
                        bins_mk2.append(bins_mk[x])
                elif var < 1.0 :
                    if hist[x] > 30 :
                        histarray.append(hist[x])
                        bins_mk2.append(bins_mk[x])     
                else :
                    if hist[x] > 40 :
                        histarray.append(hist[x])
                        bins_mk2.append(bins_mk[x])
                    
        
        #print(max(max(bins_mk2)))
        for x in range(0,len(distarray)) :
            if bins_mk2 != [] :
                if min(min(bins_mk2)) <= distarray[x] <= max(max(bins_mk2)) :
                    distarray2.append(distarray[x])   
                    
            else :
                distarray2 = prevdistarray
                
        prevdistarray = distarray2

        #print(Variance,average)
        #print(histarray)
        #print(distarray2)         
        #print('-------------------------'+str(count)+'------------------------')
        img = cv2.add(view,mask0)  
        distmedian = np.median(distarray2)
        distarray_median.append(distmedian)
        
        if startlabel is not None and finishlabel is not None :
                if startlabel <= count <= finishlabel :
                        if distmedian > 1 :
                            disttotal += distmedian
            
        #print(Filename,'--frame_num :' ,count ,'value(cm) :' ,distmedian , 'distance(m) : ',disttotal/100)
        distance_print = "dist: %.2f m"%(disttotal/100)
        #print(distarray)
        cv2.putText(img, distance_print,(0,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
        
        #showMotionVectors(distarray,distarray2,distmedian,bins)
        #print(Filename,'--frame_num :' ,count ,'value(cm) :' ,distmedian , 'distance(m) : ',disttotal/100)
        #print('분산 :',var,' 평균 : ',average)
        prevgray = framegray
        #cv2.imshow('orginal',img)
        #cv2.imwrite('img/'+Filename+'/'+Filename+'-frame'+str(count)+'.jpg',img)
       
        k = cv2.waitKey(30)
        if k == 27:
            break
        count += 1
        
    #line(distarray_median)
    print(Filename,'--frame_num :' ,count ,'value(cm) :' ,distmedian , 'distance(m) : ',disttotal/100)
    cap.release()
    cv2.destroyAllWindows()
distcalc('in_5_normal_1',35,201)
distcalc('in_5_normal_2',60,223)
distcalc('in_5_normal_3',49,214)
distcalc('in_5_fast_1',34,149)
distcalc('in_5_fast_2',35,153)
distcalc('in_5_fast_3',36,150)
distcalc('in_5_slow_1',55,268)
distcalc('in_5_slow_2',52,259)
distcalc('in_5_slow_3',62,265)
distcalc('in_5_stop_1',32,322)
distcalc('in_5_stop_2',48,347)
distcalc('in_5_stop_3',46,352)
distcalc('in_10_normal_1',34,356)
distcalc('in_10_normal_2',44,334)
distcalc('in_10_normal_3',50,337)
distcalc('in_10_fast_1',54,262)
distcalc('in_10_fast_2',17,228)
distcalc('in_10_fast_3',39,245)
distcalc('in_10_slow_1',42,443)
distcalc('in_10_slow_2',69,448)
distcalc('in_10_slow_3',62,423)
distcalc('in_10_stop_1',38,511)
distcalc('in_10_stop_2',56,580)
distcalc('in_10_stop_3',56,567)
distcalc('out_5_normal_1',42,192)
distcalc('out_5_normal_2',71,212)
distcalc('out_5_normal_3',57,190)
distcalc('out_5_fast_1',46,163)
distcalc('out_5_fast_2',34,148)
distcalc('out_5_fast_3',41,153)
distcalc('out_5_slow_1',53,231)
distcalc('out_5_slow_2',54,219)
distcalc('out_5_slow_3',61,227)
distcalc('out_5_stop_1',38,324)
distcalc('out_5_stop_2',51,312)
distcalc('out_5_stop_3',46,246)
distcalc('out_10_normal_1',63,324)
distcalc('out_10_normal_2',50,270)
distcalc('out_10_normal_3',45,288)
distcalc('out_10_fast_1',25,228)
distcalc('out_10_fast_2',52,272)
distcalc('out_10_fast_3',53,263)
distcalc('out_10_slow_1',57,374)
distcalc('out_10_slow_2',49,332)
distcalc('out_10_slow_3',65,346)
distcalc('out_10_stop_1',40,420)
distcalc('out_10_stop_2',57,405)
distcalc('out_10_stop_3',52,378)
distcalc('out_5_shadow',36,142)