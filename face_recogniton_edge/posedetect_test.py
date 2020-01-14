# From Python
# It requires OpenCV installed for Python
"""
对姿势进行检测
"""
import sys
import cv2
import os
import numpy

# Remember to add your installation path here
# Option a
sys.path.append('../../../../python')
dir_path = os.path.dirname(os.path.realpath(__file__))

try:
    from openpose import openpose as op
    # from openpose import *
except:
    raise Exception(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')

params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["disable_blending"] = False
# Ensure you point to the correct path where models are located
params["default_model_folder"] = dir_path + "/../../../../../models/"
# Construct OpenPose object allocates GPU memory
openpose = op.OpenPose(params)
print("进入姿势识别")

# 姿势识别函数
def gesture_recognize(keypoints,flags):  # define function to recognize gesture
    # flags = numpy.zeros((1, 4))  # initial count of each gesture are all 0
    v_56 = keypoints[6, 0:2] - keypoints[5, 0:2]  # left shoulder and arm
    # v_67 = keypoints[7, 0:2] - keypoints[6, 0:2]  # left arm and hand
    v_15 = keypoints[5, 0:2] - keypoints[1, 0:2]  # neck and left shoulder
    v_12 = keypoints[2, 0:2] - keypoints[1, 0:2]  # neck and right shoulder
    v_23 = keypoints[3, 0:2] - keypoints[2, 0:2]  # right shoulder and arm
    v_17 = keypoints[7, 0:2] - keypoints[1, 0:2]  # neck and left hand
    v_14 = keypoints[4, 0:2] - keypoints[1, 0:2]  # neck and right hand
    # normalize vector
    nv_56 = v_56 / numpy.linalg.norm(v_56, ord=1)
    # nv_67 = v_67 / numpy.linalg.norm(v_67, ord=1)
    nv_15 = v_15 / numpy.linalg.norm(v_15, ord=1)
    nv_12 = v_12 / numpy.linalg.norm(v_12, ord=1)
    nv_23 = v_23 / numpy.linalg.norm(v_23, ord=1)
    nv_17 = v_17 / numpy.linalg.norm(v_17, ord=1)
    nv_14 = v_14 / numpy.linalg.norm(v_14, ord=1)
    dv_15_56 = numpy.vdot(nv_15, nv_56)
    # print('dv_15_56 = %f' %dv_15_56)
    dv_12_23 = numpy.vdot(nv_12, nv_23)
    # print('dv_12_23 = %f' % dv_12_23)
    dv_v_17 = numpy.vdot([0,1], nv_17)  # dot product of positive vertical line and the v_17,
                                        # if we lift left hand it should be positive
    dv_v_14 = numpy.vdot([0, 1], nv_14) # dot product of positive vertical line and the v_14,
                                        # if we lift right hand it should be positive
    # if (dv_15_56 >= 0 and dv_15_56 <= 0.7 and dv_v_17 > 0 and dv_v_14 <= 0):  # the angle btw arm_shoulder
    if (dv_12_23 >= 0 and dv_12_23 <= 0.7 and dv_v_14 > 0 and dv_v_17 <= 0):
        flags[0, 0] = flags[0, 0] + 1  # and the neck_arm is 90-135 degree
    # elif (dv_12_23 >= 0 and dv_12_23 <= 0.7 and dv_v_14 > 0 and dv_v_17 <= 0):  # the angle btw arm_shoulder
    elif (dv_15_56 >= 0 and dv_15_56 <= 0.7 and dv_v_17 > 0 and dv_v_14 <= 0):
        flags[0, 1] = flags[0, 1] + 1  # and the neck_arm is 90-135 degree
    print(flags)
    return flags

# 所需要从边上调用的姿势识别，输入识别图像地址，返回是否满足
def pose_detect(img_path, save_path):
    # Read new image
    img = cv2.imread(img_path)
    # Output keypoints and the image with the human skeleton blended on it
    keypoints, output_image = openpose.forward(img, True)
    """
    # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
    print(keypoints)
    # Display the image
    cv2.imshow("output", output_image)
    cv2.waitKey(0)
    """

    # 存储检测结果
    cv2.imwrite(save_path, output_image)
    # output_image.save("/test_img_1.jpg")

    # 暂时认为有指定姿势返回大于1，没有返回0
    temp_pose_id = temp(keypoints)
    if temp_pose_id >= 1:  # 大于1 说明有指定姿势被识别（暂定 左手或右手抬起）
        print("temp_pose_id = %s" % temp_pose_id)
        return True
    else:
        return False



#  调用姿势识别 判断姿势
def temp(keypoints):
    flags = numpy.zeros((1, 4))
    if numpy.size(keypoints) != 0:
        keypoints = keypoints[0, :, :]
        flags = gesture_recognize(keypoints, flags)  # let each image compare with keypoints transfered vector and
        if(flags[0, 0] >= 1):  # 识别出左手抬起
            return 1
        if (flags[0, 1] >= 1):  # 识别出右手抬起
            return 2
        else:
            return 0
    else:
        return 0
