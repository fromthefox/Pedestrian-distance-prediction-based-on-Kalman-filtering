import cv2

def get_persons_pos(frame, cascade_classifier = cv2.CascadeClassifier('/Users/a111/Downloads/haarcascade_fullbody.xml')):
    """
    获取帧画面里的所有行人的位置
    :param frame: 画面帧
    :return: 所有行人位置的列表
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 检测人体
    bodies = cascade_classifier.detectMultiScale(gray, 1.1, 4)
    # bodies的格式是:(x, y, w, h)
    return bodies

def get_person_distance(h):
    """
    单目视频测距
    :param h:行人高度
    :return: 行人距离
    """
    foc = 1700.0
    # 镜头焦距, 已经测量出来
    real_hight_person = 66.9
    # 行人高度
    dis_inch = (real_hight_person * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/100
    return dis_m

def get_target_pos(first_frame, MIN_DIS = 5):
    """
    开始跟踪的第一帧
    :param first_frame:帧画面
    :return: 需要跟踪的目标的位置:默认选择所有行人中距离最近的那个进行跟踪
    if return == 'nobody' means nobody has been detected
    if return == 0, [] means somebody has been detected but no one is under the dangerous distance
    else means someone is dangerous and must be attentioned
    """
    bodies = get_persons_pos(first_frame)
    min_dis = MIN_DIS
    # 设置最近的检测阈值:如果距离大于5m,则认为没有进入危险距离;不予跟踪
    target_pos = []
    dis = 0
    if len(bodies) == 0:
        return 0, 'nobody'
    else:
        for (x, y, w, h) in bodies:
            dis = get_person_distance(h)
            if dis < min_dis:
                min_dis = dis
                target_pos = (x, y, w, h)
        return min_dis, target_pos
