#!/usr/bin/env python
# coding: utf-8

# ## Python PDR 구현

# ### 1. Kalman Filter

# In[100]:


import numpy as np
from numpy.linalg import inv

class Kf:
    def __init__(self, X, P):
        self._X = X
        self.__P = P

    def _filter(self, H, Q, R, Z, A, XP):
        ## PP = A * P * A.T + Q
        pp = np.matmul(A, self.__P)
        pp = np.matmul(pp, A.T)
        pp = np.add(pp, Q)

        ## K = PP * H.T * Inv( H * PP * H.T + R )
        k = np.matmul(pp, H.T)
        tmp = np.matmul(H, pp)
        tmp = np.matmul(tmp, H.T)
        tmp = np.add(tmp, R)
        tmp = inv(tmp)
        k = np.matmul(k, tmp)

        ## X = XP + ( K * ( Z - ( H * XP ) ) )
        tmp = np.matmul(H, XP)
        tmp = np.subtract(Z, tmp)
        tmp = np.matmul(k, tmp)
        self._X = np.add(XP, tmp)

        ## P = PP - ( K * ( H * PP ) )
        tmp = np.matmul(H, pp)
        tmp = np.matmul(k, tmp)
        self.__P = np.subtract(pp, tmp)


# ### 2. Step Distance - Acceler Noise Smoothing (Kalman Filter)

# In[101]:


import numpy as np
from numpy.linalg import inv

class AccNoiseKf(Kf):
    def __init__(self):
        X = np.zeros((2, 1))
        P = np.array([[1., 0.],
                          [0., 1.]])
        Kf.__init__(self, X, P)

        self.__H = np.array([[0., 1.]])
        self.__Q = np.array([[0.1, 0.],
                             [0., 0.1]])
        self.__R = np.array([[100.]])

    def filter(self, accScl, dt):
        Z = np.array([[accScl]])
        A = np.array([[1., dt],
                             [0., 1.]])

        ## XP = A * X
        XP = np.matmul(A, self._X)

        self._filter(self.__H, self.__Q, self.__R, Z, A, XP)

        return self._X[1, 0]


# ### 3. Step distance

# In[102]:


import math

class StepDistance:
    # acc_avg: 별도 테스트로 측정된 사용자 가속도 평균
    # acc_var: 별도 테스트로 측정된 사용자 가속도 분산
    # constant: 사용자 키(M) * K
    def __init__(self, acc_avg=0.02586745, acc_var=0.017735777, constant=0.47*1.74):
        self.__oldElapsedTimeMillis = 0
        self.__accmax = [0., 0.]         #최대 가속도
        self.__accmin = 0.               #최소 가속도
        self.__thmax = acc_var / 2.      #(걸음으로 인식하는) 최대 임계값
        self.__thmin = -self.__thmax       #('') 최소 임계값
        self.__constant = constant
        self.__acc_peak = [0., 0., 0.]
        self.__stepst = False            #이동 중
        self.__stepst1 = False           #3축 가속도가 최소점에서 위로 올라가는 중
        self.__stepst2 = False           #3축 가속도가 최소점을 지나 올라간 상태
        self.__gt2_tmp = 0
        self.__acc_avg = acc_avg
        self.__step_no = 1                #걸음idx ( 걸음 수 = 걸음idx - 1 )
        self.__kf = AccNoiseKf()

    # acc: 가속도 센서 [ 0, 0, 0 ]
    # elapsedTimeMillis: 가속도 센서가 측정된 시간
    def calculate(self, acc, elapsedTimeMillis):
        dt = (elapsedTimeMillis - self.__oldElapsedTimeMillis) * 0.001
        self.__oldElapsedTimeMillis = elapsedTimeMillis

        # 3축 가속도 크기
        accScl = math.sqrt(acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]) / 9.8 - 1.0

        # 가속도 smoothing KF
        self.__acc_peak[2] = self.__acc_peak[1]
        self.__acc_peak[1] = self.__acc_peak[0]
        self.__acc_peak[0] = self.__kf.filter(accScl, dt) - self.__acc_avg

        # 발걸음 이동했는지 판단
        if self.__acc_peak[0] >= self.__thmax:
            self.__stepst = True

        steplen = 0

        if self.__stepst:
            # 걸음 인식
            tmp = (self.__acc_peak[1] - self.__acc_peak[2]) * (self.__acc_peak[0] - self.__acc_peak[1])

            ## Start state
            if tmp < 0. and self.__acc_peak[1] >= self.__thmax and self.__acc_peak[1] > self.__accmax[0]:
                ### peak[2]       : 최대점을 지나기 직전
                ### peak[1]       : 최대점. (thmax, accmax[0] 보다 크다.)
                ### peak[0] (현재): 최대점을 지난 직후

                ### 최대 가속도 재설정 (적응형)
                self.__accmax[0] = self.__acc_peak[1]
                if self.__accmax[0] > 4.:
                    self.__accmax[0] = 4.

            ## state 2
            if tmp < 0. and self.__acc_peak[1] <= self.__thmin:
                ### peak[2]       : 최소점을 지나기 직전
                ### peak[1]       : 최소점. (thmin 보다 작다.)
                ### peak[0] (현재): 최소점을 지난 직후
                
                self.__stepst2 = True

                ### 최소 가속도 재설정 (적응형)
                if self.__acc_peak[1] < self.__accmin:
                    self.__accmin = self.__acc_peak[1]
                
                ### 최대 임계값 재설정 (적응형)
                if self.__step_no < 3: #3걸음 이내
                    self.__thmax = 1. / 3. * self.__accmax[0]
                else:
                    self.__thmax = 1. / 3. * 0.5 * (self.__accmax[0] + self.__accmax[1])

            ## state 1
            if self.__acc_peak[1] < 0. and self.__acc_peak[0] > 0. and self.__stepst2:
                ### peak[1]        : <= 1
                ### peak[0]        : >= 1
                
                self.__stepst1 = True

            ## step state
            if self.__stepst1:
                self.__step_no += 1

                step_time = elapsedTimeMillis * 0.001 - self.__gt2_tmp
                steplen = self.__constant * math.sqrt(1. / step_time) * math.sqrt(math.sqrt(self.__accmax[0] - self.__accmin))
                self.__gt2_tmp = elapsedTimeMillis * 0.001

                self.__accmax[1] = self.__accmax[0]
                self.__accmax[0] = 0.
                self.__accmin = 0.
                self.__stepst = False
                self.__stepst1 = False
                self.__stepst2 = False

        return steplen

    def getStepCount(self):
        return self.__step_no - 1


# ### 4. Azimuth - Euler Angle (Extended Kalman Filter)

# In[103]:


import numpy as np
from numpy.linalg import inv
import math

class EulerAngleEKf(Kf):
    def __init__(self):
        X = np.zeros((3, 1))
        P = np.array([[5., 0., 0.],
                      [0., 5., 0.],
                      [0., 0., 5.]])
        Kf.__init__(self, X, P)

        self.__H = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        self.__Q = np.array([[0.0001, 0., 0.],
                             [0., 0.0001, 0.],
                             [0., 0., 0.1]])
        self.__R = np.array([[10., 0.],
                             [0., 10.]])

        # orient[0]: Phi, Roll, X (세로) 축
        # orient[1]: Theta, Pitch, Y (가로) 축
        # orient[2]: Psi, Yaw, Z (수직) 축
        self.orient = np.array([0., 0., 0.])

        # orient의 변화량
        self.dorient = np.array([0., 0., 0.])

    def filter(self, acc, gyro, dt):
        # 가속도로 오일러각 계산
        Z = self.__calcPhiTheta(acc)
        A = self.__adamJcob(gyro, dt)
        XP = self.__predict(gyro, dt)

        self._filter(self.__H, self.__Q, self.__R, Z, A, XP)

        # 결과 저장 & 반환
        self.dorient = self.orient #tmp
        self.orient = self._X[:, 0]
        self.dorient = self.orient - self.dorient

        return self.orient

    def __signum(self, n):
        if n == 0.0 or n == None:
            return n
        return math.copysign(1.0, n)

    def __calcPhiTheta(self, acc):
        G = 9.8066 #중력가속도

        acctmp = acc[0] / G
        theta = -math.sin(acctmp) if abs(acctmp) < 1.0 else math.pi / 2 * 0.99 * self.__signum(-acctmp)
        
        acctmp = acc[1] / (G * math.cos(theta))
        phi = math.asin(acctmp) if abs(acctmp) < 1.0 else math.pi / 2 * 0.99 * self.__signum(acctmp)

        return np.array([[phi], [theta]])

    def __fixRange(self, v, minv, maxv):
        if v < minv:
            return minv
        elif v > maxv:
            return maxv
        return v

    # 추정한 값을 편미분
    def __adamJcob(self, gyro, dt):
        phi = self.__fixRange(self._X[0, 0], -89.5, 89.5)
        theta = self.__fixRange(self._X[1, 0], -89.5, 89.5)
        p = gyro[0]
        q = gyro[1]
        r = gyro[2]

        A = np.array([[q * math.cos(phi) * math.tan(theta) - r * math.sin(phi) * math.tan(theta),
                       q * math.sin(phi) / (math.cos(theta) * math.cos(theta)) + r * math.cos(phi) / math.pow(math.cos(theta), 2),
                       0],
                      [-q * math.sin(phi) - r * math.cos(phi),
                       0,
                       0],
                      [q * math.cos(phi) / math.cos(theta) - r * math.sin(phi) / math.cos(theta),
                       q * math.sin(phi) / math.cos(theta) * math.tan(theta) + r * math.cos(phi) / math.cos(theta) * math.tan(theta),
                       0]])
        return np.add(A * dt, np.array([[1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 1]]))
    
    # 값 추정
    def __predict(self, gyro, dt):
        phi = self.__fixRange(self._X[0, 0], -89.5, 89.5)
        theta = self.__fixRange(self._X[1, 0], -89.5, 89.5)
        p = gyro[0]
        q = gyro[1]
        r = gyro[2]

        # 회전변환... 물체의 각속도(gyro)를 오일러 각속도로 변환
        x = np.array([[p + q * math.sin(phi) * math.tan(theta) + r * math.cos(phi) * math.tan(theta)],
                      [q * math.cos(phi) - r * math.sin(phi)],
                      [q * math.sin(phi) / math.cos(theta) + r * math.cos(phi) / math.cos(theta)]])
        return np.add(self._X, x * dt)


# ### 5. Azimuth - Magnetic Azimuth Smoothing (Kalman Filter)

# In[104]:


import numpy as np
from numpy.linalg import inv

class MagAzimuthNoiseKf(Kf):
    def __init__(self):
        X = np.zeros((2, 1))
        P = np.array([[1., 0.],
                      [0., 1.]])
        Kf.__init__(self, X, P)

        self.__H = np.array([[1., 0.]])
        self.__Q = np.array([[0.001, 0.],
                             [0., 0.001]])
        self.__R = np.array([[0.5]])

    def filter(self, azimuth, dt):
        Z = np.array([[azimuth]])
        A = np.array([[1., dt],
                      [0., 1.]])

        ## XP = A * X
        XP = np.matmul(A, self._X)

        self._filter(self.__H, self.__Q, self.__R, Z, A, XP)
        
        return self._X[0, 0]


# ### 6. Azimuth - dYaw Smoothing (Kalman Filter)

# In[105]:


import numpy as np
from numpy.linalg import inv

class DYawNoiseKf(Kf):
    def __init__(self):
        X = np.zeros((2, 1))
        P = np.array([[1., 0.],
                      [0., 1.]])
        Kf.__init__(self, X, P)

        self.__H = np.array([[0., 1.]])
        self.__Q = np.array([[0.01, 0.],
                             [0., 0.01]])
        self.__R = np.array([[10.]])

    def filter(self, dYaw):
        Z = np.array([[dYaw]])
        A = np.array([[1., 1.],
                      [0., 1.]])

        ## XP = A * X
        XP = np.matmul(A, self._X)

        self._filter(self.__H, self.__Q, self.__R, Z, A, XP)
        
        return self._X[1, 0]


# ### 7. Azimuth

# In[106]:


class Azimuth:
    def __init__(self, startAngle=0):
        self.__oldElapsedTimeMillis = 0
        self.__TH_TURN = 0.0005
        self.__TH_VAR = 1.5
        self.__eulerAngleEkf = EulerAngleEKf()
        self.__magAzimuthNoiseKf = MagAzimuthNoiseKf()
        self.__dYawNoiseKf = DYawNoiseKf()
        self.__lastMagAzimuth = None
        self.__startAngle = startAngle
        self.__curAngle = startAngle
        self.__eulerYaw = startAngle    #bkYaw
        self.__magYaw = startAngle      #bkPsi
        self.__psiErr = startAngle

    # acc: 가속도 센서 [ 0, 0, 0 ]
    # gyro: 자이로 센서 [ 0, 0, 0 ]
    # mag: 지자기 센서 [ 0, 0, 0 ]
    # elapsedTimeMillis: 가속도 센서가 측정된 시간
    # declination: 편각
    def calculate(self, acc, gyro, mag, elapsedTimeMillis, declination):
        dt = (elapsedTimeMillis - self.__oldElapsedTimeMillis) * 0.001
        self.__oldElapsedTimeMillis = elapsedTimeMillis
        
        # 지자기센서 방위각 계산
        magAzimuth = self.__calcMagAzimuth(mag[0], mag[1])
        
        # 지자기센서 방위각 범위조절
        if self.__lastMagAzimuth is not None:
            if magAzimuth - self.__lastMagAzimuth > 1.7 * math.pi:
                magAzimuth -= (2 * math.pi)
            elif magAzimuth - self.__lastMagAzimuth < -1.7 * math.pi:
                magAzimuth += (2 * math.pi)
        self.__lastMagAzimuth = magAzimuth
        
        # 지자기센서 방위각 smoothing KF
        magAzimuth = self.__magAzimuthNoiseKf.filter(magAzimuth, dt) + declination
        
        # 가속도센서 오일러각 추정 KF
        orient = self.__eulerAngleEkf.filter(acc, gyro, dt)
        yaw = 2 * math.pi - orient[2] + self.__psiErr
        dYaw = self.__eulerAngleEkf.dorient[2]
        
        # dYaw smoothin KF
        dYaw = self.__dYawNoiseKf.filter(dYaw)
        
        # 방위각 추정. dYaw가 TH_TURN 임계값을 넘으면 회전 인식
        if abs(dYaw) > self.__TH_TURN:
            ## Euler Yaw 변화량
            diffEulerYaw = yaw - self.__magYaw
            diffEulerYaw = self.__toPlusMinusPi(diffEulerYaw)
            ## Mag Yaw 변화량
            diffMagYaw = magAzimuth - self.__magYaw
            diffMagYaw = self.__toPlusMinusPi(diffMagYaw)

            if (diffMagYaw < self.__TH_VAR * diffEulerYaw and diffMagYaw > (2. - self.__TH_VAR) * diffEulerYaw) or                    (diffMagYaw > self.__TH_VAR * diffEulerYaw and diffMagYaw < (2. - self.__TH_VAR) * diffEulerYaw):
                ## Euler Yaw 변화량과 Mag Yaw 변화량이 비슷함
                self.__curAngle = magAzimuth
            else:
                ## 틀리면 방금 구한 magAzimuth를 버리고 이전 magAzimuth와 Euler Yaw 변화량을 더한 각도를 사용
                self.__curAngle = self.__magYaw + yaw - self.__eulerYaw

            ## 결과 저장
            self.__eulerYaw = yaw
            self.__magYaw = self.__curAngle
        
        return self.__to2PiCircular(self.__curAngle)

    def __calcMagAzimuth(self, x, y):
        return -math.atan2(x, y)

    def __toPlusMinusPi(self, radian):
        TWO_PI = 2 * math.pi
        while radian < -math.pi:
            radian += TWO_PI
        while radian > math.pi:
            radian -= TWO_PI
        return radian

    def __to2PiCircular(self, radian):
        TWO_PI = 2 * math.pi
        while radian < 0:
            radian += TWO_PI
        while radian >= TWO_PI:
            radian -= TWO_PI
        return radian


# ### 8. PDR

# In[109]:


import math

class Pdr:
    def __init__(self, stepDistance=None):
        self.__azimuth = Azimuth()
        self.__yaw = 0

        self.__stepDistance = stepDistance if stepDistance else StepDistance()
        self.__totalDistance = 0
        self.__accumulatedPos = [0., 0.]
        self.__movedPos = [0., 0.]

    def calculate(self, acc, gyro, mag, elapsedTimeMillis, declination):
        self.__yaw = self.__azimuth.calculate(acc, gyro, mag, elapsedTimeMillis, declination)

        stepLength = self.__stepDistance.calculate(acc, elapsedTimeMillis)
        self.__movedPos[0] = stepLength * math.cos(self.__yaw)
        self.__movedPos[1] = stepLength * math.sin(self.__yaw)
        self.__accumulatedPos[0] += self.__movedPos[0]
        self.__accumulatedPos[1] += self.__movedPos[1]
        self.__totalDistance += stepLength

    def getYaw(self):
        return self.__yaw

    def getStepCount(self):
        return self.__stepDistance.getStepCount()

    def getMovedX(self):
        return self.__movedPos[0]

    def getMovedY(self):
        return self.__movedPos[1]

    def getAccumulatedX(self):
        return self.__accumulatedPos[0]

    def getAccumulatedY(self):
        return self.__accumulatedPos[1]
    
    def getTotalDistance(self):
        return self.__totalDistance


# ### 9. Test

# In[108]:


# import matplotlib.pyplot as plt

# TEST_DIR = 'test'

# pdr = Pdr()
# totDists = []
# dists = []

# file = open('%s/Optical Flow 190312/fast 17step 1.txt' % TEST_DIR)

# txtLine = file.readline()
# while txtLine:
#     row = txtLine.split()
#     time = float(row[0]) * 1000
#     acc = [float(i) for i in row[1:4]]
#     mag = [float(i) for i in row[4:7]]
#     gyro = [float(i) for i in row[7:10]]

#     pdr.calculate(acc, gyro, mag, time, 0)
#     totDists.append([time * 0.001, pdr.getAccumulatedX()])
#     dists.append([time * 0.001, pdr.getMovedX()])
#     txtLine = file.readline()

# totDists = np.array(totDists)
# dists = np.array(dists)

# print(totDists[-1][1], 'm')
# print(pdr.getStepCount(), 'steps')

# plt.figure(figsize=(10,4))
# plt.subplot(121)
# plt.plot(totDists[:,0], totDists[:,1])
# plt.subplot(122)
# plt.plot(dists[:,0], dists[:,1])
# plt.show()

# file.close()

