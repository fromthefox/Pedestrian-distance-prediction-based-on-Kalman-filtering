import cv2
import numpy as np
import AIT1000_walkman


def box_bounding_to_box_center(box_bounding):
    """
    用于把输入的边界值转换为中心点的xy坐标以及box的高度和宽度
    box_bounding:[left, top, right, bottom]
    :param box_bounding: 边界值
    :return: 中心点的xy坐标以及box的高度和宽度
    """
    center_x = (int(box_bounding[0]) + int(box_bounding[2])) / 2
    center_y = (int(box_bounding[1]) + int(box_bounding[3])) / 2
    box_width = (int(box_bounding[2]) - int(box_bounding[0]))
    box_height = (int(box_bounding[3]) - int(box_bounding[1]))
    return (center_x,center_y,box_width,box_height)


def box_center_to_box_bounding(box_center):
    """
    用于把输入的中心点xy位置以及宽高转化为边界像素值
    box_center:[center_x,center_y,width,height]
    :param box_bounding: 边界值
    :return: 四个边界值
    """
    x1 = box_center[0] - box_center[2]//2
    y1 = box_center[1] - box_center[3]//2
    x2 = box_center[0] + box_center[2] // 2
    y2 = box_center[1] + box_center[3] // 2
    return [x1, y1, x2, y2]

def plot_one_box(box_bounding, img, color=(0, 200, 0), target=False):
    """
    用于绘制边框
    :param box_bounding:边框的边界值
    :param img:绘制的图像
    :param color:颜色
    :param target:是否是我们正在跟踪的目标,如果是,则绘制为红色
    :return: NONE
    """
    right_bottom = (int(box_bounding[0]), int(box_bounding[1]))
    # xy1是右边框和下边框
    left_top = (int(box_bounding[2]), int(box_bounding[3]))
    # xy2是左边框和上边框
    if target:
        color = (0, 0, 255)
    cv2.rectangle(img, right_bottom, left_top, color, 1, cv2.LINE_AA)  # filled
    # 调用cv2的矩形绘制函数绘制矩形,cv2.LINE_AA是绘制的线的样式,1表示线的磅数

def updata_trace_list(box_center, trace_list, max_list_len=50):
    """
    用于更新轨迹列表,这个trace_list是在draw_trace函数里面使用的,用于绘制轨迹
    :param box_center:box的中心
    :param trace_list:一系列box的center,包括之前的很多个帧的跟踪目标的box_center
    :param max_list_len:最大列表长度,直接决定了绘制的轨迹的长度
    :return: NONE
    """
    if len(trace_list) <= max_list_len:
        trace_list.append(box_center)
        # 如果长度不够,也就是在跟踪刚开始的阶段,直接往里面加就行
    else:
        trace_list.pop(0)
        trace_list.append(box_center)
        # 后面的阶段,为了保证轨迹为定长度,因此先pop后append
    return trace_list

def draw_trace(img, trace_list):
    """
    用于绘制轨迹列表
    :param img: 图像
    :param trace_list: 轨迹列表
    :return: NONE
    """
    for i, item in enumerate(trace_list):
        # i == index
        # item = trace_list[i]
        if i < 1:
            continue
        cv2.line(img,
                 (item[0], item[1]), (trace_list[i - 1][0], trace_list[i - 1][1]),
                 (255, 255, 0), 3, cv2.LINE_AA)
        # 调用cv2的line函数进行画线
        # parameters:img + start_point + end_point + color + width + line_style

def cal_iou(box1, box2):
    """
    计算两个box的iou,iou越大说明两个box的重合度越高,重合度最高的box就认为是我们的观测值,用于后面的KF
    :param box1: box1
    :param box2: box2
    :return: iou
    """
    # box1 是第一个box的左上右下
    # box2 是第二个box的左上右下
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    # print('--------')
    # print(x1max, x1min, y1max, y1min)
    # print(x2max, x2min, y2max, y2min)
    # print('--------')
    # 计算两个框的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    # +1.是框的偏移
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交部分的坐标
    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)
    # 这里的xmin,xmax,ymin,ymax构成了一个box,这个box是box1和box2的重合部分


    inter_h = max(ymax - ymin + 1, 0)
    inter_w = max(xmax - xmin + 1, 0)
    # 计算box1 & box2的重合部分的宽高(如果有重合部分的话)

    intersection = inter_h * inter_w
    # 重合部分的面积
    union = s1 + s2 - intersection
    # union == 总面积 - 重合部分的面积,也就是没有重合的总面积

    # 计算iou
    iou = intersection / union
    # print(iou)
    # iou == box1和box2相交部分的面积 / box1和box2不相交部分的面积
    # iou越高，就说明两个box的契合度越高
    return iou

def cal_distance(box1, box2):
    """
    用于计算两个box的中心点的距离
    :param box1: box1
    :param box2: box2
    :return: distance
    """
    center1 = ((box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2)
    center2 = ((box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2)
    dis = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    return dis


def Kalman_walkman(last_best_estimated_pos_center, last_best_estimated_P, frame, cascade_classifier='/Users/a111/Downloads/haarcascade_fullbody.xml'):
    """
    :param last_best_estimated_pos: 上一帧的最优估计的目标位置,box_bounding形式
    :param last_best_estimated_P: 上一帧的最优估计的矩阵P
    :param frame: 视频帧
    :return: new_best_estimated_pos, new_best_estimated_P这一帧的最优估计的目标位置
    """
    '''
    这里需要先用yolo或者其他方法识别出frame中的所有行人框位置，
    做成一个list，假设为walkman_pos_list
    '''
    last_best_estimated_pos_bounding = box_center_to_box_bounding(last_best_estimated_pos_center)
    walkman_pos_list = []
    max_iou = 0.3
    # 在这里设置IOU的阈值
    target_matched_judgement = False
    last_best_estimated_status = np.array([[last_best_estimated_pos_center[0],
                                          last_best_estimated_pos_center[1],
                                          last_best_estimated_pos_center[2],
                                          last_best_estimated_pos_center[3],
                                          0,
                                          0]]).T
    observated_status = np.array([[0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0]]).T
    # 初始化X和Z
    # 这里的walkman_pos_list就是我们需要调用算法得到的,frame中所有的行人的位置,bounding形式
    walkman_pos_list = AIT1000_walkman.get_persons_pos(frame)
    for index, box_pos in enumerate(walkman_pos_list):
        box_pos = box_center_to_box_bounding(box_pos)
        iou = cal_iou(last_best_estimated_pos_bounding, box_pos)
        # 把这一帧里的所有行人框和上一帧的最优估计进行契合度的计算，
        # 把iou最高的box作为这一帧的观测值

        if iou >= max_iou:
            target_pos_bounding = box_pos
            # 记录box位置
            max_iou = iou
            # 更新max_iou
            target_matched_judgement = True

    if target_matched_judgement:
        # 如果匹配上，则进行kalman filter的预测
        # 先计算dx,dy
        target_pos_center = box_bounding_to_box_center(target_pos_bounding)
        dx = target_pos_center[0] - last_best_estimated_pos_center[0]
        dy = target_pos_center[1] - last_best_estimated_pos_center[1]
        # 补全X和Z
        last_best_estimated_status[4], last_best_estimated_status[5] = dx, dy
        # print(target_pos_center)
        observated_status[0:4] = np.array(target_pos_center).reshape(4, 1)
        observated_status[4], observated_status[5] = dx, dy

        X_last = last_best_estimated_status
        Z = observated_status
        # 也就是观测值直接都用这一帧的检测值即可

        # 其中H、R、B都是不变的，直接定义就可以（或者传递过来，这里图简单，就直接定义了）
        H = np.eye(6)
        R = np.eye(6)
        P_last = last_best_estimated_P
        Q = np.eye(6) * 0.1
        B = None
        # 状态转移矩阵
        A = np.array([[1, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 1],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])
        '''
            状态转移矩阵，上一时刻的状态转移到当前时刻
            x2 = x1 + dx
            y2 = y1 + dy
            w2 = w1 认为矩形框高度宽度不变
            h2 = h1
            dx2 = dx1 
            dy2 = dy1 认为目标匀速运动?
        '''
        # -----进行先验估计-----------------
        X_prior = np.dot(A, X_last)
        # box_prior = box_center_to_box_bounding(X_prior[0:4])
        # plot_one_box(box_prior, frame, color=(0, 0, 0), target=False)
        # -----计算状态估计协方差矩阵P--------
        P_prior = np.dot(np.dot(A, P_last), A.T) + Q
        # ------计算卡尔曼增益---------------------
        k1 = np.dot(P_prior, H.T)
        k2 = np.dot(np.dot(H, P_prior), H.T) + R
        K = np.dot(k1, np.linalg.inv(k2))
        # --------------后验估计------------
        # X_posterior_1 = Z - np.dot(H, X_prior)
        X_best_estimated = X_prior + np.dot(K, Z - np.dot(H, X_prior))
        box_posterior = box_center_to_box_bounding(X_best_estimated[0:4])
        # plot_one_box(box_posterior, frame, color=(255, 255, 255), target=False)
        # ---------更新状态估计协方差矩阵P-----
        # P_posterior_1 = np.eye(6) - np.dot(K, H)
        I = np.eye(6)
        P_best_estimated = np.dot(I-np.dot(K,H), P_prior)
        return box_posterior, P_best_estimated, False
        # 有匹配上的行人,FIRST_FLAG置为False,表示下一帧不需要重新选择目标
    else:
        # 如果没有匹配上，则认为这一帧的最优估计与上一帧相比，保持不变
        # 没有匹配上的情况对应的就是目标丢失,目标丢失则需要获取一个新的追踪目标,调用算法获取危险距离内的最近目标即可
        # P也进行初始化
        new_best_estimated_P = np.eye(6)
        target_dis, target_pos = AIT1000_walkman.get_target_pos(frame)
        if target_pos == 'nobody' or target_pos == []:
            # 如果没有需要追踪的target:则直接返回上一帧的目标位置
            new_best_estimated_pos = last_best_estimated_pos_bounding
        else:
            # 如果有需要追踪的目标,就把需要追踪的目标作为下一次检测的目标位置
            new_best_estimated_pos = box_center_to_box_bounding(target_pos)
        return new_best_estimated_pos, new_best_estimated_P, True
        # 没有匹配上目标,因此下一帧需要重新选择目标





"""
@9.4 @Fox
目前存在的问题:
1. 如何识别图片中每一帧的所有行人的位置
    1.1. YOLO:识别率高,算法复杂度也高一些?
    1.2. 通过其他算法?找找看
2. 如何确定我们要跟踪的目标
    2.1. 计算Box面积,自动选择面积最大的那个Box进行跟踪预测
    2.2. 多目标跟踪预测
3. 如何把我们的YOLO的预测结果和我们的预警系统结合起来?也就是,什么情况下我们认为是危险情况?
4. 如何获取新的一帧?@Line 251

@9.6 @Fox
解决9.4的问题:
1. 目前使用HOG+OpenCV处理
2. & 3. 用单目测距地方法进行处理
    2. 对于first_frame的距离最近的行人(且该行人小于危险阈值)进行跟踪;如果跟踪丢失,则换一个最近的其他的距离小于危险阈值的目标进行跟踪
    3. 行人目标 < 5m: 警戒
       行人目标 < 2m: 进行报警
3. 直接把新的一个frame传入即可 
"""