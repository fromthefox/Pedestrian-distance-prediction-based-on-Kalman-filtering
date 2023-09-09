"""
@9.6 @Fox
完整的Kalman滤波 + 单目测距 + 行人检测预警系统
"""
import AIT1000_kalman
import AIT1000_walkman
import numpy as np
import cv2

if __name__ == '__main__':
    P = np.eye(6)
    # initialize P
    # cascade_classifier = cv2.CascadeClassifier('/Users/a111/Downloads/haarcascade_fullbody.xml')
    # load walkman detector
    cap = cv2.VideoCapture('./test_video.mp4')
    # load video
    FIRST_FLAG = True
    # FLAG置为True
    FRAME_CNT = 0
    MIN_DIS = 10
    # 预警距离
    pos = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        FRAME_CNT += 1
        # load frame
        '''
        FIRST_FLAG == True means we need a new 'pos'
        FIRST_FLAG == False means we don't need a new 'pos'
        '''
        if FRAME_CNT % 4 == 0:
            if FIRST_FLAG:
                dis_camara, pos = AIT1000_walkman.get_target_pos(frame, MIN_DIS)
                # dis_camara == 0 means nobody needs to be tracked
                # so dis_camara will less than 5
                if dis_camara < MIN_DIS and dis_camara != 0:
                    # dis > 0 means someone needs to been tracked
                    pos, P, FIRST_FLAG = AIT1000_kalman.Kalman_walkman(pos, P, frame)
                    # FIRST_FLAG == Ture means someone in the frame has matched with the pos we give
                    pos_center = AIT1000_kalman.box_bounding_to_box_center(pos)
                    dis_predict = AIT1000_walkman.get_person_distance(pos_center[3])
                    if dis_predict < 5:
                        print('报警')
                    else:
                        print('警戒')
                else:
                    # dis_camara在这个范围之外,说明没有检测到行人或者没有需要追踪的目标,直接跳过这一帧即可
                    continue
            else:
                # 如果FIRST_FLAG = False,说明下一帧不需要匹配目标,因此我们直接对下一帧进行常规预测即可;
                actually_dis = AIT1000_walkman.get_person_distance(pos[3])
                print('---------------')
                print('this_frame_actually_dis', actually_dis)
                pos, P, FIRST_FLAG = AIT1000_kalman.Kalman_walkman(pos, P, frame)
                print('👇🏻')
                pos_center = AIT1000_kalman.box_bounding_to_box_center(pos)
                dis_predict = AIT1000_walkman.get_person_distance(pos_center[3])
                print('next_frame_predicted_dis', dis_predict)
                print('---------------')
                if dis_predict < 5:
                    print('报警')
                else:
                    print('警戒')
'''
@9.6 @Fox
对于第一帧而言:
    没有pos可以传递到kalman函数中,因此需要一个pos;
    因此需要调用get_target_pos来获取目标:
        如果返回dis == 0说明没有需要跟踪的目标,直接continue,不改变FIRST_FLAG
        如果返回dis != 0说明有需要跟踪的目标,将该目标记录,直接传入Kalman中;
            此时的kalman返回结果一定是FIRST_FLAG == True(相当于此时是自己和自己匹配,一定可以匹配上,所以一定会返回FIRST_FLAG)
            根据Kalman Filter的公式可以知道,dx == dy == 0,那么输出的结果就等于输入;
            也就是说,此时的kalman返回结果一定就是我们用get_target_pos得到的跟踪目标的原始位置
            因此此时的dis_predicted == dis, 根据dis就可以直接判断是否需要报警
对于第一帧之后的第n帧而言:
    如果第n帧的walkman能够和第n-1帧的最优估计进行匹配:
        根据第n-1帧的最优估计,以及第n帧的观测值,可以预测第n帧的最优估计
        (也就是 kalman (best_estimated_pos_of_frame_n-1, best_estimated_P_of_frame_n-1, frame_n)
        对第n帧的最优估计进行单目测距,如果距离小于危险值则报警;如果距离小于阈值但是大于危险值则attention
        继续循环
    如果第n帧的walkman不能够和第n-1帧的最优估计进行匹配(也就是我们所说的目标丢失)
        那么FIRST_FLAG == True,且此时kalman返回的pos就是第n-1帧的最优估计的pos:
            我们需要调用get_target_pos来获取目标:
                如果返回dis == 0说明没有需要跟踪的目标,pos仍然保持为n-1帧的最优估计的pos,然后continue到下一帧,也就是第n+1帧;
                如果返回dis != 0说明有需要跟踪的目标,则将pos更新为该目标的pos,然后传入Kalman中:
                    此时的情况和上面一样,一定可以找到匹配项,然后将FIRST_FLAG更新为False;
                    且kalman得到的结果一定就是get_target_pos的结果,dis_predicted也就是get_target_pos的dis;
                    然后就会回到@Line 932        

'''
'''
@9.8 @Fox
基本完成
'''
