import pypdr
import math
import matplotlib.pyplot as plt

def calctime(Filename):
    sensorFile = open('./sensor video2/'+Filename+'.txt', 'rt', encoding='utf-8')
    count = 0
    if sensorFile:
        # PDR 유틸 클래스 생성
        pdr = pypdr.Pdr(pypdr.StepDistance(0.02586745, 0.017735777, 0.47*1.74))
    
        # PDR 결과를 그래프로 보여주기 위해 사용하는 배열
        times = []          # 센서 측정시간 배열
        yaws = []           # PDR Yaw 배열
        accumulatedXs = []  # PDR 누적거리 배열
        accumulatedYs = []  # PDR 누적거리 배열
        movedXs = []        # PDR 보폭 배열
        movedYs = []        # PDR 보폭 배열
        # 센서파일 내용 로드
        txtLine = sensorFile.readline()
        while txtLine:
            count += 1
            row = txtLine.split()
            time = float(row[0]) * 1000          # 센서 측정시간 (millis)
            acc = [float(i) for i in row[1:4]]   # 가속도센서
            mag = [float(i) for i in row[4:7]]   # 지자기센서
            gyro = [float(i) for i in row[7:10]] # 자이로센서
    
            # PDR 계산
            pdr.calculate(acc, gyro, mag, time, 0)
    
            # PDR 계산결과
            yaw = pdr.getYaw()                     # Yaw (Radian)
            totalDistance = pdr.getTotalDistance() # 현재까지 총 거리
            accumulatedX = pdr.getAccumulatedX()   # 누적거리 X
            accumulatedY = pdr.getAccumulatedY()   # 누적거리 Y
            movedX = pdr.getMovedX()               # 보폭 X
            movedY = pdr.getMovedY()               # 보폭 Y
            stepCount = pdr.getStepCount()         # 걸음수
            
            
            #print(yaw)
            #print(math.cos(yaw),count)
            # 그래프를 그리기 위해 PDR 계산결과 저장
            times.append(time * 0.001)
            yaws.append(math.cos(yaw))
            accumulatedXs.append(accumulatedX)
            accumulatedYs.append(accumulatedY)
            movedXs.append(movedX)
            movedYs.append(movedY)
    
            txtLine = sensorFile.readline()
        sensorFile.close()
    return time

def calcyaw(Filename) :
    sensorFile = open('./sensor video2/'+Filename+'.txt', 'rt', encoding='utf-8')
    count = 0
    if sensorFile:
        # PDR 유틸 클래스 생성
        pdr = pypdr.Pdr(pypdr.StepDistance(0.02586745, 0.017735777, 0.47*1.74))
    
        # PDR 결과를 그래프로 보여주기 위해 사용하는 배열
        times = []          # 센서 측정시간 배열
        yaws = []           # PDR Yaw 배열
        accumulatedXs = []  # PDR 누적거리 배열
        accumulatedYs = []  # PDR 누적거리 배열
        movedXs = []        # PDR 보폭 배열
        movedYs = []        # PDR 보폭 배열
        # 센서파일 내용 로드
        txtLine = sensorFile.readline()
        while txtLine:
            count += 1
            row = txtLine.split()
            time = float(row[0]) * 1000          # 센서 측정시간 (millis)
            acc = [float(i) for i in row[1:4]]   # 가속도센서
            mag = [float(i) for i in row[4:7]]   # 지자기센서
            gyro = [float(i) for i in row[7:10]] # 자이로센서
    
            # PDR 계산
            pdr.calculate(acc, gyro, mag, time, 0)
    
            # PDR 계산결과
            yaw = pdr.getYaw()                     # Yaw (Radian)
            totalDistance = pdr.getTotalDistance() # 현재까지 총 거리
            accumulatedX = pdr.getAccumulatedX()   # 누적거리 X
            accumulatedY = pdr.getAccumulatedY()   # 누적거리 Y
            movedX = pdr.getMovedX()               # 보폭 X
            movedY = pdr.getMovedY()               # 보폭 Y
            stepCount = pdr.getStepCount()         # 걸음수
            
            
            #print(yaw)
            #print(math.cos(yaw),count)
            # 그래프를 그리기 위해 PDR 계산결과 저장
            times.append(time * 0.001)
            yaws.append(math.cos(yaw))
            accumulatedXs.append(accumulatedX)
            accumulatedYs.append(accumulatedY)
            movedXs.append(movedX)
            movedYs.append(movedY)
    
            txtLine = sensorFile.readline()
        sensorFile.close()
    return yaw
    
#        ### Pdr 결과, 그래프 출력
#        print('총 이동거리=%fm (%d 걸음)' % (pdr.getTotalDistance(), pdr.getStepCount()))
#    
#        plt.figure(figsize=(12, 3))
#    
#        plt.subplot(121)
#        plt.title('total distance each time')
#        plt.xlabel('time')
#        plt.ylabel('distance(m)')
#        plt.plot(times, accumulatedXs)
#    
#        plt.subplot(122)
#        plt.title('distance each time')
#        plt.xlabel('time')
#        plt.plot(times, movedXs)
#        plt.show()
#       
#        plt.figure(figsize=(12, 3))
#    
#        plt.subplot(121)
#        plt.title('yaw each time')
#        plt.xlabel('time')
#        plt.ylabel('radian')
#        #plt.ylim(-.08, .08)
#        plt.plot(times, yaws)
#    
#        plt.subplot(122)
#        plt.title('tracking')
#        plt.xlabel('x (m)')
#        plt.ylabel('y (m)')
#        plt.plot(accumulatedXs, accumulatedYs)
#        plt.show()
        
#calcyaw('in_5_normal_1')
#calcyaw('in_5_normal_2')
#calcyaw('in_5_normal_3')
#calcyaw('in_5_fast_1')
#calcyaw('in_5_fast_2')
#calcyaw('in_5_fast_3')
#calcyaw('in_5_slow_1')
#calcyaw('in_5_slow_2')
#calcyaw('in_5_slow_3')
#calcyaw('in_5_stop_1')
#calcyaw('in_5_stop_2')
#calcyaw('in_5_stop_3')
#calcyaw('in_10_normal_1')
#calcyaw('in_10_normal_2')
#calcyaw('in_10_normal_3')
#calcyaw('in_10_fast_1')
#calcyaw('in_10_fast_2')
#calcyaw('in_10_fast_3')
#calcyaw('in_10_slow_1')
#calcyaw('in_10_slow_2')
#calcyaw('in_10_slow_3')
#calcyaw('in_10_stop_1')
#calcyaw('in_10_stop_2')
#calcyaw('in_10_stop_3')
#calcyaw('out_5_normal_1')
#calcyaw('out_5_normal_2')
#calcyaw('out_5_normal_3')
#calcyaw('out_5_fast_1')
#calcyaw('out_5_fast_2')
#calcyaw('out_5_fast_3')
#calcyaw('out_5_slow_1')
#calcyaw('out_5_slow_2')
#calcyaw('out_5_slow_3')
#calcyaw('out_5_stop_1')
#calcyaw('out_5_stop_2')
#calcyaw('out_5_stop_3')
#calcyaw('out_10_normal_1')
#calcyaw('out_10_normal_2')
#calcyaw('out_10_normal_3')
#calcyaw('out_10_fast_1')
#calcyaw('out_10_fast_2')
#calcyaw('out_10_fast_3',)
#calcyaw('out_10_slow_1')
#calcyaw('out_10_slow_2')
#calcyaw('out_10_slow_3')
#calcyaw('out_10_stop_1')
#calcyaw('out_10_stop_2')
#calcyaw('out_10_stop_3')