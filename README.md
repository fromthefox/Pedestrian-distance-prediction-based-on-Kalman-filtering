# 基于Kalman滤波的行人距离检测-python

# 基于Kalman滤波的行人距离检测-python
Pedestrian distance detection based on Kalman filtering

# September 9, 2023 @Fox

## 算法特点

利用Kalman Filter的预测性,结合摄像头单目测距地方法进行行人位置预警,可以用在自动驾驶小车等领域.

## 使用方法

1. 将需要检测的video路径替换AIT1000_Walkman_Kalman.py中的cap = cv2.VideoCapture(‘path/to/your/video’)
2. 选择需要的阈值：包括报警距离和警告距离
3. 运行AIT1000_Walkman_Kalman.py文件

## 拓展使用

1. 可以根据需求进行标框
2. 可以根据需求分装函数
3. ……

## 基本技术

1. 通过cv2.HOG-SVM进行单张图片中的行人进行检测；
2. 通过AIT_1000.get_person_distance实现单目摄像头测距；
3. 通过Kalman Filter实现目标跟踪和预测；

## 具体思路

1. **对于第一帧而言:**
没有pos可以传递到kalman函数中,因此需要一个pos;
因此需要调用get_target_pos来获取目标:
    1. 如果返回**dis_camara == MIN_DIS**,说明没有需要跟踪的目标,直接continue,不改变FIRST_FLAG
    2. 如果返回**dis_camara ≠ MIN_DIS**,说明有需要跟踪的目标,将该目标记录,直接传入Kalman中;此时的kalman返回结果一定是FIRST_FLAG == True(相当于此时是自己和自己匹配,一定可以匹配上,所以一定会返回FIRST_FLAG)
    根据Kalman Filter的公式可以知道,dx == dy == 0,那么输出的结果就等于输入;也就是说,此时的kalman返回结果一定就是我们用get_target_pos得到的跟踪目标的原始位置,因此此时的dis_predicted == dis_camara, 根据dis就可以直接判断是否需要报警
2. **对于第一帧之后的第n帧而言:**
    1. **如果第n帧的walkman能够和第n-1帧的最优估计进行匹配:**
    根据第n-1帧的最优估计,以及第n帧的观测值,可以预测第n帧的最优估计
    (也就是 kalman (best_estimated_pos_of_frame_n-1, best_estimated_P_of_frame_n-1, frame_n)
    对第n帧的最优估计进行单目测距,如果距离小于危险值则报警;如果距离小于阈值但是大于危险值则警戒
    继续循环
    2. **如果第n帧的walkman不能够和第n-1帧的最优估计进行匹配(也就是我们所说的目标丢失)**
    那么FIRST_FLAG == True,且此时kalman返回的pos就是第n-1帧的最优估计的pos:
    我们需要调用get_target_pos来获取目标:
    如果返回dis == 0说明没有需要跟踪的目标,pos仍然保持为n-1帧的最优估计的pos,然后continue到下一帧,也就是第n+1帧;
    如果返回dis != 0说明有需要跟踪的目标,则将pos更新为该目标的pos,然后传入Kalman中:
    此时的情况和上面一样,一定可以找到匹配项,然后将FIRST_FLAG更新为False;
    且kalman得到的结果一定就是get_target_pos的结果,dis_predicted也就是get_target_pos的dis;
    然后就会回到2.a.