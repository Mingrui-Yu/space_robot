import pybullet as p
import time
import numpy as np
import pybullet_data
import cv2
from scipy.linalg import block_diag



class Environment(object):
    def __init__(self):
        self.project_dir = '/home/mingrui/Mingrui/Homework/space_robot/'

        self.physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

        self.sim_rate = 240
        self.sim_count = 0

        self.grasp_joint_angles = None

        codec = cv2.VideoWriter_fourcc('M', 'P', 'E', "G")
        fps = 30
        frameSize = (int(1280), int(720))
        self.video1 = cv2.VideoWriter(self.project_dir + "results/video1.avi", codec, fps, frameSize)
        self.video2 = cv2.VideoWriter(self.project_dir + "results/video2.avi", codec, fps, frameSize)

        self.base_force_in_baseMove = []
        self.base_pos_in_baseMove = []
        self.target_pos_in_baseMove = []
        self.target_avel_in_baseMove = []
        self.base_avel_in_baseMove = []

        self.end_pos_in_moveToGraspPoint = []
        self.target_pos_in_moveToGraspPoint = []

        self.base_torque_in_stopMotion = []
        self.end_6DoF_force_in_stopMotion = []
        self.joint_torque_in_stopMotion = []
        self.joint_angle_in_stopMotion = []

        self.base_force_in_moveToWorkPose = []
        self.joint_torque_in_moveToWorkPose = []

    
    # -----------------------------------------------------------------------------------------------
    def camera1(self):
        # params for realsense d435
        height = int(720)
        width = int(1280)
        fov = 42 # degree
        aspect = 16/9

        camera_view_matrix = p.computeViewMatrixFromYawPitchRoll(
                        cameraTargetPosition=[0, 7, 0],
                        distance=20,
                        yaw=60, pitch=-30, roll=0,
                        upAxisIndex=2)

        camera_projection_matrix = p.computeProjectionMatrixFOV(
                    fov=fov,               # 摄像头的视线夹角
                    aspect=aspect,  # 画幅宽/高
                    nearVal=1,            # 摄像头焦距下限
                    farVal=100              # 摄像头能看上限
                    )                                                     

        camera = p.getCameraImage(width=width, height=height, viewMatrix=camera_view_matrix, projectionMatrix=camera_projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

        image_bgr = cv2.cvtColor(np.array(camera[2])[:, :, 0:3], cv2.COLOR_RGB2BGR)
        # cv2.imshow("camera1", image_bgr)
        # cv2.waitKey(1)

        self.video1.write(image_bgr)

    
    # -----------------------------------------------------------------------------------------------
    def camera2(self):
        # params for realsense d435
        height = int(720)
        width = int(1280)
        fov = 42 # degree
        aspect = 16/9

        camera_view_matrix = p.computeViewMatrixFromYawPitchRoll(
                        cameraTargetPosition=[0, 7, 0],
                        distance=20,
                        yaw=120, pitch=30, roll=0,
                        upAxisIndex=2)

        camera_projection_matrix = p.computeProjectionMatrixFOV(
                    fov=fov,               # 摄像头的视线夹角
                    aspect=aspect,  # 画幅宽/高
                    nearVal=1,            # 摄像头焦距下限
                    farVal=100              # 摄像头能看上限
                    )                                                     

        camera = p.getCameraImage(width=width, height=height, viewMatrix=camera_view_matrix, projectionMatrix=camera_projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

        image_bgr = cv2.cvtColor(np.array(camera[2])[:, :, 0:3], cv2.COLOR_RGB2BGR)
        # cv2.imshow("camera2", image_bgr)
        # cv2.waitKey(1)

        self.video2.write(image_bgr)


    # -----------------------------------------------------
    def initiateScene(self):
        # 设置GUI视角
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=60,
                             cameraPitch=-30, cameraTargetPosition=[5, 3, 5])
        # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0,
        #                      cameraPitch=-80, cameraTargetPosition=[0, 2, 6])

        
        self.initiateSpaceRobot()
        self.initiateGipper()
        self.initiateCutter()
        self.initiateTargetSatellite()
        
        
    # -----------------------------------------------------------------------------------------------
    def initiateSpaceRobot(self):
        # load robot
        start_pos = [0, 0 ,0]
        start_ori = p.getQuaternionFromEuler([0,0,0])
        # flags = p.URDF_USE_INERTIA_FROM_FILE  # | p.URDF_MERGE_FIXED_LINKS
        self.space_robot_id = p.loadURDF(self.project_dir + "pybullet/urdf/space_robot_with_mass_6.urdf",start_pos, start_ori)

        # 清除默认的damping
        p.changeDynamics(self.space_robot_id, -1, linearDamping=1e-3, angularDamping=1e-3)

        # 初始化，提取关节参数
        self.initiateSpaceRobotJointIds()

        # 初始化机械臂
        self.initiateArms()


    # -----------------------------------------------------------------------------------------------
    def initiateGipper(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.space_robot_id)

        self.gripper_initial_local_pos = [0.4, 1.8, -1.5]
        self.gripper_initial_local_quat = p.getQuaternionFromEuler([np.pi/2, 0, 0])
        start_pos, start_quat = p.multiplyTransforms(base_pos, base_quat, 
                                                                self.gripper_initial_local_pos, self.gripper_initial_local_quat) # np.pi/2, 0, np.pi

        self.gripper_id = p.loadURDF(self.project_dir + "pybullet/urdf/gripper3.urdf", start_pos, start_quat)

        # 新建fixed 约束
        gripper_cm_pos, gripper_cm_quat = p.getBasePositionAndOrientation(self.gripper_id)
        joint_pos = base_pos
        joint_quat = base_quat
        base_pos_inv, base_quat_inv = p.invertTransform(base_pos, base_quat)
        joint_pos_in_parent, joint_quat_in_parent = p.multiplyTransforms(base_pos_inv, base_quat_inv, joint_pos, joint_quat)
        gripper_pos_inv, gripper_quat_inv = p.invertTransform(gripper_cm_pos, gripper_cm_quat)
        joint_pos_in_child, joint_quat_in_child = p.multiplyTransforms(gripper_pos_inv, gripper_quat_inv, joint_pos, joint_quat)
        
        self.base_gripper_constraint_id = p.createConstraint(parentBodyUniqueId=self.space_robot_id, parentLinkIndex=-1, childBodyUniqueId=self.gripper_id, childLinkIndex=-1,
            jointType=p.JOINT_FIXED, jointAxis=[0, 1, 0], 
            parentFramePosition=joint_pos_in_parent, childFramePosition=joint_pos_in_child,
            parentFrameOrientation=joint_quat_in_parent, childFrameOrientation=joint_quat_in_child)

        p.setJointMotorControl2(self.gripper_id, 0, p.POSITION_CONTROL, targetPosition=0, force=10000)
        p.setJointMotorControl2(self.gripper_id, 1, p.POSITION_CONTROL, targetPosition=0, force=10000)
        p.setJointMotorControl2(self.gripper_id, 2, p.POSITION_CONTROL, targetPosition=0, force=10000)


    # -----------------------------------------------------------------------------------------------
    def initiateCutter(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.space_robot_id)

        self.cutter_initial_local_pos = [-0.4, 1.8, -1.5]
        self.cutter_initial_local_quat = p.getQuaternionFromEuler([np.pi/2, 0, 0])
        start_pos, start_quat = p.multiplyTransforms(base_pos, base_quat, 
                                                                self.cutter_initial_local_pos, self.cutter_initial_local_quat ) # np.pi/2, 0, np.pi
        self.cutter_id = p.loadURDF(self.project_dir + "pybullet/urdf/cutter.urdf",start_pos, start_quat)

        # 新建fixed 约束
        cutter_cm_pos, cutter_cm_quat = p.getBasePositionAndOrientation(self.cutter_id)
        cutter_cm_pos = np.array(cutter_cm_pos)
        cutter_cm_quat = np.array(cutter_cm_quat)

        joint_pos = base_pos
        joint_quat = base_quat
        base_pos_inv, base_quat_inv = p.invertTransform(base_pos, base_quat)
        joint_pos_in_parent, joint_quat_in_parent = p.multiplyTransforms(base_pos_inv, base_quat_inv, joint_pos, joint_quat)
        cutter_pos_inv, cutter_quat_inv = p.invertTransform(cutter_cm_pos, cutter_cm_quat)
        joint_pos_in_child, joint_quat_in_child = p.multiplyTransforms(cutter_pos_inv, cutter_quat_inv, joint_pos, joint_quat)
        
        self.base_cutter_constraint_id = p.createConstraint(parentBodyUniqueId=self.space_robot_id, parentLinkIndex=-1, childBodyUniqueId=self.cutter_id, childLinkIndex=-1,
            jointType=p.JOINT_FIXED, jointAxis=[0, 1, 0], 
            parentFramePosition=joint_pos_in_parent, childFramePosition=joint_pos_in_child,
            parentFrameOrientation=joint_quat_in_parent, childFrameOrientation=joint_quat_in_child)


    # -----------------------------------------------------------------------------------------------
    def initiateSpaceRobotJointIds(self):
        self.joint_num = p.getNumJoints(self.space_robot_id)

        # 打印joint info，并创建一个字典，存储joint_name-joint_id / joint_name-joint_DoF_id
        self.joint_name_id_dict = {}
        self.joint_name_DoF_id_dict = {}
        joint_DoF_id = 0
        for i in range(self.joint_num):
            joint_info =  p.getJointInfo(self.space_robot_id, i)
            joint_id = joint_info[0]
            joint_name= joint_info[1].decode('UTF-8')
            joint_type = joint_info[2]
            print('joint id: ', joint_id, ', joint name: ', joint_name ) #
            self.joint_name_id_dict[joint_name] = joint_id
            if joint_type != p.JOINT_FIXED: # 目前只适用于1自由度的关节
                self.joint_name_DoF_id_dict[joint_name] = joint_DoF_id
                joint_DoF_id += 1

        # 需要的joint_name
        self.arm_1_joint_name = ['joint11', 'joint12', 'joint13', 'joint14', 'joint15', 'joint16', 'joint17']
        self.arm_2_joint_name = ['joint21', 'joint22', 'joint23', 'joint24', 'joint25', 'joint26', 'joint27']

        # 根据需要的joint_name提取相应的joint_id和joint_DoF_id
        self.arm_1_id = []
        self.arm_2_id = []
        self.arm_1_joint_DoF_id = []
        self.arm_2_joint_DoF_id = []
        for name in self.arm_1_joint_name:
            self.arm_1_id.append(self.joint_name_id_dict[name])
            self.arm_1_joint_DoF_id.append(self.joint_name_DoF_id_dict[name])
        for name in self.arm_2_joint_name:
            self.arm_2_id.append(self.joint_name_id_dict[name])
            self.arm_2_joint_DoF_id.append(self.joint_name_DoF_id_dict[name])


    # -----------------------------------------------------------------------------------------------
    def initiateArms(self):
        # arm 0
        # 初始化关节角度
        p.resetJointState(self.space_robot_id, self.arm_1_id[0],  targetValue=1/2*np.pi)
        p.resetJointState(self.space_robot_id, self.arm_1_id[1],  targetValue=0*np.pi)
        p.resetJointState(self.space_robot_id, self.arm_1_id[2],  targetValue=1/2*np.pi)
        p.resetJointState(self.space_robot_id, self.arm_1_id[3],  targetValue=-1*np.pi)
        p.resetJointState(self.space_robot_id, self.arm_1_id[4],  targetValue=1*np.pi)
        p.resetJointState(self.space_robot_id, self.arm_1_id[5],  targetValue=0*np.pi)
        p.resetJointState(self.space_robot_id, self.arm_1_id[6],  targetValue=0*np.pi)
        # 初始化位置控制器
        for id in self.arm_1_id:
            current_position = p.getJointState(self.space_robot_id, id)[0]
            p.setJointMotorControl2(self.space_robot_id, id, p.POSITION_CONTROL, targetPosition=current_position, force=10000) 

        # arm 1
        # 初始化关节角度
        p.resetJointState(self.space_robot_id, self.arm_2_id[0],  targetValue=-1/2*np.pi)
        p.resetJointState(self.space_robot_id, self.arm_2_id[1],  targetValue=0*np.pi)
        p.resetJointState(self.space_robot_id, self.arm_2_id[2],  targetValue=1/2*np.pi)
        p.resetJointState(self.space_robot_id, self.arm_2_id[3],  targetValue=-1*np.pi)
        p.resetJointState(self.space_robot_id, self.arm_2_id[4],  targetValue=1*np.pi)
        p.resetJointState(self.space_robot_id, self.arm_2_id[5],  targetValue=0*np.pi)
        p.resetJointState(self.space_robot_id, self.arm_2_id[6],  targetValue=0*np.pi)
        # 初始化位置控制器
        for id in self.arm_2_id:
            current_position = p.getJointState(self.space_robot_id, id)[0]
            p.setJointMotorControl2(self.space_robot_id, id, p.POSITION_CONTROL, targetPosition=current_position, force=10000) 


    # -----------------------------------------------------------------------------------------------
    def initiateTargetSatellite(self):
        # load 目标卫星
        start_pos = [0, 10, 0]
        start_ori = p.getQuaternionFromEuler([0,0,0])
        # flags = p.URDF_USE_INERTIA_FROM_FILE
        self.target_satellite_id = p.loadURDF(self.project_dir + "pybullet/urdf/xin_nuo_4.urdf", start_pos, start_ori)

        # 清除默认的damping
        p.changeDynamics(self.target_satellite_id, -1, linearDamping=1e-3, angularDamping=1e-3)

        self.target_satellite_grasp_point_offset = [0.47, -1.79842914118884 - 0.40, 0]

        self.target_satellite_joint_num = p.getNumJoints(self.target_satellite_id)
        # 打印joint info，并创建一个字典，存储joint_name-joint_id
        self.target_satellite_joint_name_id_dict = {}
        for i in range(self.target_satellite_joint_num):
            joint_info =  p.getJointInfo(self.target_satellite_id, i)
            joint_id = joint_info[0]
            joint_name= joint_info[1].decode('UTF-8')
            joint_type = joint_info[2]
            print('joint id: ', joint_id, ', joint name: ', joint_name ) #
            self.target_satellite_joint_name_id_dict[joint_name] = joint_id
        
        # 未展开的状态
        p.resetJointState(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_tian_xian1'],  targetValue=1/2*np.pi)
        p.resetJointState(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_tian_xian2'],  targetValue=0)

        p.resetJointState(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_zhuanzhou1'],  targetValue=0)
        p.resetJointState(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_lianjiegan1'],  targetValue=-1/2*np.pi)
        p.resetJointState(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_1fanban1'],  targetValue=-np.pi)
        p.resetJointState(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_1fanban2'],  targetValue=-np.pi)
        p.resetJointState(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_1fanban3'],  targetValue=-1/2*np.pi)

        p.resetJointState(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_zhuanzhou_2'],  targetValue=np.pi)
        p.resetJointState(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_lianjiegan2'],  targetValue=1/2*np.pi)
        p.resetJointState(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_2fanban1'],  targetValue=np.pi)
        p.resetJointState(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_2fanban2'],  targetValue=-np.pi)
        p.resetJointState(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_2fanban3'],  targetValue=1/2*np.pi)

        # 初始化位置控制器，保持住关节位置
        for id in range(self.target_satellite_joint_num):
            current_position = p.getJointState(self.target_satellite_id, id)[0]
            p.setJointMotorControl2(self.target_satellite_id, id, p.POSITION_CONTROL, targetPosition=current_position, force=10000) 
        
 
    # -----------------------------------------------------------------------------------------------
    def openTargetSatellite(self, object):
        max_vel = 0.05
        if object == 'antenna_2':
            p.setJointMotorControl2(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_tian_xian2'], p.POSITION_CONTROL, targetPosition=-np.pi/2, maxVelocity=max_vel) 
        
        elif object == 'panel_1-4':
            p.setJointMotorControl2(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_lianjiegan1'],  p.POSITION_CONTROL, targetPosition=0, maxVelocity=max_vel)
            p.setJointMotorControl2(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_1fanban1'],  p.POSITION_CONTROL, targetPosition=0, maxVelocity=max_vel*2)
            p.setJointMotorControl2(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_1fanban2'],  p.POSITION_CONTROL, targetPosition=0, maxVelocity=max_vel*2)
            p.setJointMotorControl2(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_1fanban3'],  p.POSITION_CONTROL, targetPosition=0, maxVelocity=max_vel)

        elif object == 'panel_5-8':
            p.setJointMotorControl2(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_lianjiegan2'], p.POSITION_CONTROL, targetPosition=0, maxVelocity=max_vel) 
            p.setJointMotorControl2(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_2fanban1'], p.POSITION_CONTROL, targetPosition=0, maxVelocity=max_vel*2)
            p.setJointMotorControl2(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_2fanban2'], p.POSITION_CONTROL, targetPosition=0, maxVelocity=max_vel*2)
            p.setJointMotorControl2(self.target_satellite_id, self.target_satellite_joint_name_id_dict['joint_2fanban3'], p.POSITION_CONTROL, targetPosition=0, maxVelocity=max_vel)


    # -----------------------------------------------------------------------------------------------
    def getSpaceRobotJointAngles(self):
        joint_angles = [0] * self.joint_num
        joint_states = p.getJointStates(self.space_robot_id, list(range(self.joint_num)))
        for i in range(self.joint_num):
            joint_angles[i] = joint_states[i][0]
        return joint_angles

    # -----------------------------------------------------------------------------------------------
    def getSpaceRobotDoFJointAnglesAndVelocityies(self):
        joint_angles = []
        joint_vels = []
        joint_states = p.getJointStates(self.space_robot_id, list(range(self.joint_num)))
        for i in range(self.joint_num):
            if p.getJointInfo(self.space_robot_id, i)[2] != p.JOINT_FIXED:
                joint_angles.append(joint_states[i][0])
                joint_vels.append(joint_states[i][1])
        return joint_angles, joint_vels


    # -----------------------------------------------------------------------------------------------
    def arm1ToInitialWorkConfiguration(self, max_vel=0.5):
        current_angles = np.array(self.getSpaceRobotJointAngles())[self.arm_1_id]
        target_angles = np.array([1/2*np.pi, 0, 1/3*np.pi, -2/3*np.pi, 1/3*np.pi, 0, 0])
        scales = np.abs(target_angles - current_angles)
        if np.linalg.norm(scales) < 1e-2:
            return True

        scales /= np.max(scales)
        for i, id in enumerate(self.arm_1_id):
            p.setJointMotorControl2(self.space_robot_id, id, p.POSITION_CONTROL, targetPosition=target_angles[i], positionGain=0.1, maxVelocity=max_vel * scales[i], force=1000) 
        return False
    

    # -----------------------------------------------------------------------------------------------
    def arm2ToInitialWorkConfiguration(self, max_vel=0.5):
        current_angles = np.array(self.getSpaceRobotJointAngles())[self.arm_2_id]
        target_angles = np.array([-1/2*np.pi, 0, 1/3*np.pi, -2/3*np.pi, 1/3*np.pi, 0, 0])
        scales = np.abs(target_angles - current_angles)
        if np.linalg.norm(scales) < 1e-2:
            return True

        scales /= np.max(scales)
        for i, id in enumerate(self.arm_2_id):
            p.setJointMotorControl2(self.space_robot_id, id, p.POSITION_CONTROL, targetPosition=target_angles[i], positionGain=0.1, maxVelocity=max_vel * scales[i], force=1000) 
        return False


    # -----------------------------------------------------------------------------------------------
    def pickOutGripper(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.space_robot_id)

        target_grasp_local_pos = np.array(self.gripper_initial_local_pos) + np.array([0, 0.4, 0])
        target_grasp_local_quat = self.gripper_initial_local_quat
        target_grasp_pos, target_grasp_quat = p.multiplyTransforms(base_pos, base_quat, 
                                                                target_grasp_local_pos, target_grasp_local_quat ) # np.pi/2, 0, np.pi

        # 求解逆运动学
        end_link_id = self.arm_2_id[-1]
        joint_damping = 0.1 * np.ones((len(self.arm_1_id) + len(self.arm_2_id), ))
        joint_damping[self.arm_2_id[0]] = 10
        joint_damping[self.arm_2_id[1]] = 10
        joint_damping[self.arm_2_id[5]] = 10
        joint_angles = np.array(p.calculateInverseKinematics(self.space_robot_id, end_link_id, target_grasp_pos, targetOrientation=target_grasp_quat,
            jointDamping=joint_damping.tolist() ))[self.arm_2_joint_DoF_id].tolist()

        # 控制arm运动
        current_angles = np.array(self.getSpaceRobotJointAngles())[self.arm_2_id]
        target_angles = np.array(joint_angles)
        scales = np.abs(target_angles - current_angles)
        if np.linalg.norm(scales) < 1e-3:
            return True

        max_vel = 0.5
        scales /= np.max(scales)
        for i, id in enumerate(self.arm_2_id):
            p.setJointMotorControl2(self.space_robot_id, id, p.POSITION_CONTROL, targetPosition=target_angles[i], positionGain=0.05, maxVelocity=max_vel * scales[i], force=1000) 
        return False


    # -----------------------------------------------------------------------------------------------
    def pickGripper(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.space_robot_id)
        # 计算期望位姿
        target_grasp_local_pos = np.array(self.gripper_initial_local_pos) + np.array([0, 0.03, 0])
        target_grasp_local_quat = self.gripper_initial_local_quat
        target_grasp_pos, target_grasp_quat = p.multiplyTransforms(base_pos, base_quat, 
                                                                target_grasp_local_pos, target_grasp_local_quat ) # np.pi/2, 0, np.pi

        # 计算jacobian矩阵
        current_joint_angles, current_joint_vel = self.getSpaceRobotDoFJointAnglesAndVelocityies()
        linear_jacobian, angular_jacobian = p.calculateJacobian(self.space_robot_id, self.arm_2_id[-1], localPosition=[0,0,0], objPositions=current_joint_angles, 
                objVelocities=[0]*len(current_joint_angles), objAccelerations=[0]*len(current_joint_angles))

        whole_jacobian = np.concatenate([np.array(linear_jacobian), np.array(angular_jacobian)], axis=0)
        jacobian_in_base = whole_jacobian[:, 6 + np.array(self.arm_2_joint_DoF_id)] # 前6位是基座漂浮的6个自由度
        R_base_in_world = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)
        jacobian_in_world = block_diag(R_base_in_world, R_base_in_world) @ jacobian_in_base

        current_end_cm_pos = (p.getLinkState(self.space_robot_id, linkIndex=self.arm_2_id[-1])[0])
        current_end_cm_quat = (p.getLinkState(self.space_robot_id, linkIndex=self.arm_2_id[-1])[1])

        delta_end_pos = np.array(target_grasp_pos) - np.array(current_end_cm_pos)
        delta_end_quat = p.getDifferenceQuaternion(current_end_cm_quat, target_grasp_quat)
        axis, angle = p.getAxisAngleFromQuaternion(delta_end_quat)
        delta_end_ori = np.array(axis) * angle
        delta_end_pose = np.concatenate([delta_end_pos.reshape(-1, 1), delta_end_ori.reshape(-1, 1)], axis=0)

        # 计算关节角速度
        k = 1.0
        desired_end_vel = k * delta_end_pose
        joint_vel = k * np.linalg.pinv(jacobian_in_world) @ desired_end_vel
        joint_vel = joint_vel.reshape(-1, )
        # 控制关节角速度
        for i in range(0, len(self.arm_2_id)):
            p.setJointMotorControl2(self.space_robot_id, self.arm_2_id[i], p.VELOCITY_CONTROL, targetVelocity=joint_vel[i], force=100)

        # 若完成
        if np.linalg.norm(np.array(current_end_cm_pos) - np.array(target_grasp_pos)) < 1e-2 \
                and np.linalg.norm(np.array(current_end_cm_quat) - np.array(target_grasp_quat)) < 1e-2:

            # 清除base和gripper之间的fixed constraint
            p.removeConstraint(self.base_gripper_constraint_id)

            # 新建end 和 gripper之间的fixed constraint
            gripper_cm_pos, gripper_cm_quat = p.getBasePositionAndOrientation(self.gripper_id)
            current_end_cm_pos = p.getLinkState(self.space_robot_id, linkIndex=self.arm_2_id[-1])[0]
            current_end_cm_quat = p.getLinkState(self.space_robot_id, linkIndex=self.arm_2_id[-1])[1]

            joint_pos = current_end_cm_pos
            joint_quat = current_end_cm_quat
            current_end_cm_pos_inv, current_end_cm_quat_inv = p.invertTransform(current_end_cm_pos, current_end_cm_quat)
            joint_pos_in_parent, joint_quat_in_parent = p.multiplyTransforms(current_end_cm_pos_inv, current_end_cm_quat_inv, joint_pos, joint_quat)
            gripper_cm_pos_inv, gripper_cm_pos_quat_inv = p.invertTransform(gripper_cm_pos, gripper_cm_quat)
            joint_pos_in_child, joint_quat_in_child = p.multiplyTransforms(gripper_cm_pos_inv, gripper_cm_pos_quat_inv, joint_pos, joint_quat)
            
            self.end_gripper_constraint_id = p.createConstraint(parentBodyUniqueId=self.space_robot_id, parentLinkIndex=self.arm_2_id[-1], childBodyUniqueId=self.gripper_id, childLinkIndex=-1,
                jointType=p.JOINT_FIXED, jointAxis=[0, 1, 0], 
                parentFramePosition=joint_pos_in_parent, childFramePosition=joint_pos_in_child,
                parentFrameOrientation=joint_quat_in_parent, childFrameOrientation=joint_quat_in_child)
            return True

        return False

    
    # -----------------------------------------------------------------------------------------------
    def baseMove(self):
        robot_base_pos, robot_base_quat = p.getBasePositionAndOrientation(self.space_robot_id)
        robot_base_pos = np.array(robot_base_pos)
        robot_base_lvel, robot_base_avel = p.getBaseVelocity(self.space_robot_id)
        target_satellite_pos, target_satellite_quat = p.getBasePositionAndOrientation(self.target_satellite_id)
        target_satellite_pos = np.array(target_satellite_pos)
        target_satellite_lvel, target_satellite_avel = p.getBaseVelocity(self.target_satellite_id)
        target_satellite_lvel = np.array(target_satellite_lvel)

        # 计算线速度
        desired_robot_base_pos = np.array(target_satellite_pos) + np.array([0, -9.5, 0])
        kp = 1.0
        input_base_linear_vel =  kp * (desired_robot_base_pos - robot_base_pos)  + target_satellite_lvel
        max_base_linear_speed = 0.1
        if np.linalg.norm(input_base_linear_vel) > max_base_linear_speed:
            input_base_linear_vel = input_base_linear_vel / np.linalg.norm(input_base_linear_vel) * max_base_linear_speed

        # 计算角速度
        kp = 1.0
        delta_end_quat = p.getDifferenceQuaternion(robot_base_quat, target_satellite_quat)
        axis, angle = p.getAxisAngleFromQuaternion(delta_end_quat)
        delta_end_ori = np.array(axis) * angle
        if np.abs(robot_base_avel[1] - target_satellite_avel[1] < 1e-1):
            input_base_avel_y = kp * delta_end_ori[1] + target_satellite_avel[1]
        else:
            input_base_avel_y = target_satellite_avel[1]

        max_base_angular_speed = 0.3
        input_base_avel_y = np.clip(input_base_avel_y, -max_base_angular_speed, max_base_angular_speed)
        
        # p.resetBaseVelocity(self.space_robot_id, linearVelocity=input_base_linear_vel, angularVelocity=[0, input_base_avel_y, 0])
        force, torque = self.baseControl(desired_pos=None, desired_lvel=input_base_linear_vel, desired_quat=None, desired_avel=[0, input_base_avel_y, 0])


        self.base_force_in_baseMove.append(np.concatenate([force, torque]))
        self.base_pos_in_baseMove.append(robot_base_pos)
        self.target_pos_in_baseMove.append(target_satellite_pos)
        self.base_avel_in_baseMove.append(robot_base_avel)
        self.target_avel_in_baseMove.append(target_satellite_avel)


        # 如果位置、姿态、线速度、角速度均满足要求，则完成
        if np.linalg.norm(desired_robot_base_pos - robot_base_pos) < 1e-2 and np.abs(delta_end_ori[1]) < 1e-2\
                and  np.abs(robot_base_avel[1] - target_satellite_avel[1] < 1e-2) and np.linalg.norm(np.array(robot_base_lvel) - target_satellite_lvel) < 1e-2:
            return True

        return False


    # -----------------------------------------------------------------------------------------------
    def moveToPreGraspPose2(self):
        target_satellite_pos, target_satellite_quat = p.getBasePositionAndOrientation(self.target_satellite_id)
        base_pos, base_quat = p.getBasePositionAndOrientation(self.space_robot_id)
        current_end_cm_pos = (p.getLinkState(self.space_robot_id, linkIndex=self.arm_2_id[-1])[0])
        current_end_cm_quat = (p.getLinkState(self.space_robot_id, linkIndex=self.arm_2_id[-1])[1])

        target_grasp_pos, target_grasp_quat = p.multiplyTransforms(target_satellite_pos, target_satellite_quat, 
                                                                self.target_satellite_grasp_point_offset, p.getQuaternionFromEuler([np.pi/2, 0, np.pi]) ) # np.pi/2, 0, np.pi

        # 计算jacobian矩阵                               
        current_joint_angles, current_joint_vel = self.getSpaceRobotDoFJointAnglesAndVelocityies()
        linear_jacobian, angular_jacobian = p.calculateJacobian(self.space_robot_id, self.arm_2_id[-1], localPosition=[0,0,0], objPositions=current_joint_angles, 
                objVelocities=[0]*len(current_joint_angles), objAccelerations=[0]*len(current_joint_angles))

        whole_jacobian = np.concatenate([np.array(linear_jacobian), np.array(angular_jacobian)], axis=0)
        jacobian_in_base = whole_jacobian[:, 6 + np.array(self.arm_2_joint_DoF_id)] # 前6位是基座漂浮的6个自由度
        R_base_in_world = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)
        jacobian_in_world = block_diag(R_base_in_world, R_base_in_world) @ jacobian_in_base

        delta_end_pos = np.array(target_grasp_pos) - np.array(current_end_cm_pos)
        delta_end_quat = p.getDifferenceQuaternion(current_end_cm_quat, target_grasp_quat)
        axis, angle = p.getAxisAngleFromQuaternion(delta_end_quat)
        delta_end_ori = np.array(axis) * angle

        delta_end_pose = np.concatenate([delta_end_pos.reshape(-1, 1), delta_end_ori.reshape(-1, 1)], axis=0)
        k = 5.0
        desired_end_vel = k * delta_end_pose
        joint_vel = k * np.linalg.pinv(jacobian_in_world) @ desired_end_vel
        joint_vel = joint_vel.reshape(-1, )

        max_joint_vel = np.pi / 6
        if np.linalg.norm(joint_vel) > max_joint_vel:
            joint_vel = joint_vel / np.linalg.norm(joint_vel) * max_joint_vel

        for i in range(0, len(self.arm_2_id)):
            p.setJointMotorControl2(self.space_robot_id, self.arm_2_id[i], p.VELOCITY_CONTROL, targetVelocity=joint_vel[i], force=1000)

        # 保存数据
        self.end_pos_in_moveToGraspPoint.append(current_end_cm_pos)
        self.target_pos_in_moveToGraspPoint.append(target_grasp_pos)


        if np.linalg.norm(np.array(current_end_cm_pos) - np.array(target_grasp_pos)) < 5e-2 \
                and np.linalg.norm(np.array(current_end_cm_quat) - np.array(target_grasp_quat)) < 5e-2:
            return True
        return False


    # -----------------------------------------------------------------------------------------------
    def makeGraspConstraint(self):
        target_satellite_pos, target_satellite_quat = p.getBasePositionAndOrientation(self.target_satellite_id)
        target_satellite_pos = np.array(target_satellite_pos)
        target_satellite_quat = np.array(target_satellite_quat)
        current_end_cm_pos = p.getLinkState(self.space_robot_id, linkIndex=self.arm_2_id[-1])[0]
        current_end_cm_quat = p.getLinkState(self.space_robot_id, linkIndex=self.arm_2_id[-1])[1]

        joint_pos = current_end_cm_pos
        joint_quat = current_end_cm_quat
        current_end_cm_pos_inv, current_end_cm_quat_inv = p.invertTransform(current_end_cm_pos, current_end_cm_quat)
        joint_pos_in_parent, joint_quat_in_parent = p.multiplyTransforms(current_end_cm_pos_inv, current_end_cm_quat_inv, joint_pos, joint_quat)
        target_satellite_pos_inv, target_satellite_quat_inv = p.invertTransform(target_satellite_pos, target_satellite_quat)
        joint_pos_in_child, joint_quat_in_child = p.multiplyTransforms(target_satellite_pos_inv, target_satellite_quat_inv, joint_pos, joint_quat)
        
        self.grasp_constraint_id = p.createConstraint(parentBodyUniqueId=self.space_robot_id, parentLinkIndex=self.arm_2_id[-1], childBodyUniqueId=self.target_satellite_id, childLinkIndex=-1,
            jointType=p.JOINT_FIXED, jointAxis=[0, 1, 0], 
            parentFramePosition=joint_pos_in_parent, childFramePosition=joint_pos_in_child,
            parentFrameOrientation=joint_quat_in_parent, childFrameOrientation=joint_quat_in_child)

        current_angles = np.array(self.getSpaceRobotJointAngles())[self.arm_2_id]
        self.grasp_joint_angles = current_angles.tolist()
        # for i in range(0, len(self.arm_2_id)):
        #     p.setJointMotorControl2(self.space_robot_id, self.arm_2_id[i], p.POSITION_CONTROL, targetPosition=current_angles[i], positionGain=1, maxVelocity=10, force=100)

        p.setJointMotorControl2(self.gripper_id, 1, p.POSITION_CONTROL, targetPosition=0.7, force=1)
        p.setJointMotorControl2(self.gripper_id, 2, p.POSITION_CONTROL, targetPosition=-0.7, force=1)

        return True


    # -----------------------------------------------------------------------------------------------
    def stopMotion(self):
        flags = p.WORLD_FRAME
        desired_avel =  [0, 0, 0]
        base_lvel, base_avel = p.getBaseVelocity(self.space_robot_id)
        delta_avel = np.array(desired_avel) - np.array(base_avel)

        kv = 1000
        torque =  kv * delta_avel
        max_torque = 100
        if np.linalg.norm(torque) > max_torque:
            torque = torque / np.linalg.norm(torque) * max_torque
        p.applyExternalTorque(self.space_robot_id, -1, torqueObj=torque, flags=flags)


        # 机械臂阻抗控制器
        for id in self.arm_2_id:
            p.setJointMotorControl2(self.space_robot_id, id, p.VELOCITY_CONTROL, targetVelocity=0, force=0) 

        Dd = np.eye(7) *1000
        Kd = np.eye(7) *1000

        joint_angles, joint_vels = self.getSpaceRobotDoFJointAnglesAndVelocityies()
        desired_acceleration = np.zeros(np.array(joint_angles).shape).tolist() # 0
        tau = np.array(p.calculateInverseDynamics(self.space_robot_id, joint_angles, joint_vels, desired_acceleration, flags=1))[self.arm_2_joint_DoF_id].reshape(-1, 1)

        desired_angles = np.array(self.grasp_joint_angles).reshape(-1, 1)
        current_angles = np.array(joint_angles)[self.arm_2_joint_DoF_id].reshape(-1, 1)
        current_vels = np.array(joint_vels)[self.arm_2_joint_DoF_id].reshape(-1, 1)

        tau += Dd @ (- current_vels) + Kd @ (desired_angles - current_angles)
        tau = tau.reshape(-1, )

        max_joint_torque = 100
        tau = np.clip(tau, -max_joint_torque, max_joint_torque)
        for i, id in enumerate(self.arm_2_id):
            p.setJointMotorControl2(self.space_robot_id, id, p.TORQUE_CONTROL, force=tau[i])


        # 保存数据
        self.base_torque_in_stopMotion.append(torque)
        self.end_6DoF_force_in_stopMotion.append(p.getConstraintState(self.grasp_constraint_id))
        self.joint_torque_in_stopMotion.append(tau)
        self.joint_angle_in_stopMotion.append(current_angles)


        target_satellite_lvel, target_satellite_avel = p.getBaseVelocity(self.target_satellite_id)
        if np.linalg.norm(target_satellite_avel) < 1e-2 \
                and np.linalg.norm(desired_angles - current_angles) < 1e-2:
            return True

        return False


    # -----------------------------------------------------------------------------------------------
    def recoverBaseControlled(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.space_robot_id)
        base_lvel, base_avel = p.getBaseVelocity(self.space_robot_id)

        delta_base_quat = p.getDifferenceQuaternion(base_quat, p.getQuaternionFromEuler([0, 0, 0]))
        axis, angle = p.getAxisAngleFromQuaternion(delta_base_quat)
        delta_base_ori = np.array(axis) * angle

        kp = 0.1
        desired_avel = kp * delta_base_ori
        max_avel = 0.05
        if np.linalg.norm(desired_avel) > max_avel:
            desired_avel = desired_avel / np.linalg.norm(desired_avel) * max_avel

        self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=None, desired_avel=desired_avel)
        # p.resetBaseVelocity(self.space_robot_id, angularVelocity=desired_avel)

        # 机械臂阻抗控制器
        for id in self.arm_2_id:
            p.setJointMotorControl2(self.space_robot_id, id, p.VELOCITY_CONTROL, targetVelocity=0, force=0) 

        Dd = np.eye(7) *1000
        Kd = np.eye(7) *1000

        joint_angles, joint_vels = self.getSpaceRobotDoFJointAnglesAndVelocityies()
        desired_acceleration = np.zeros(np.array(joint_angles).shape).tolist() # 0
        tau = np.array(p.calculateInverseDynamics(self.space_robot_id, joint_angles, joint_vels, desired_acceleration, flags=1))[self.arm_2_joint_DoF_id].reshape(-1, 1)

        desired_angles = np.array(self.grasp_joint_angles).reshape(-1, 1)
        current_angles = np.array(joint_angles)[self.arm_2_joint_DoF_id].reshape(-1, 1)
        current_vels = np.array(joint_vels)[self.arm_2_joint_DoF_id].reshape(-1, 1)

        tau += Dd @ (- current_vels) + Kd @ (desired_angles - current_angles)
        tau = tau.reshape(-1, )

        max_joint_torque = 100
        for i, id in enumerate(self.arm_2_id):
            p.setJointMotorControl2(self.space_robot_id, id, p.TORQUE_CONTROL, force=np.min([max_joint_torque, tau[i]]) ) 

        if np.linalg.norm(delta_base_ori) < 1e-1 and np.linalg.norm(np.array(base_avel)) < 1e-1 \
                and np.linalg.norm(desired_angles - current_angles) < 1e-1 and np.linalg.norm(current_vels) < 1e-1:
            return True

        return False


    # -----------------------------------------------------------------------------------------------
    def moveTargetToWorkPose(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.space_robot_id)

        target_grasp_pos, target_grasp_quat = p.multiplyTransforms(base_pos, base_quat, 
                                                                [self.target_satellite_grasp_point_offset[0], 6.5, 0], p.getQuaternionFromEuler([np.pi/2, 0, np.pi]) ) # np.pi/2, 0, np.pi


        end_link_id = self.arm_2_id[-1]
        joint_damping = 0.1 * np.ones((len(self.arm_1_id) + len(self.arm_2_id), ))
        joint_damping[self.arm_2_id[0]] = 100
        desired_angles = np.array(p.calculateInverseKinematics(self.space_robot_id, end_link_id, target_grasp_pos, targetOrientation=target_grasp_quat,
            jointDamping=joint_damping.tolist() ))[self.arm_2_joint_DoF_id]

        # 机械臂阻抗控制器
        for id in self.arm_2_id:
            p.setJointMotorControl2(self.space_robot_id, id, p.VELOCITY_CONTROL, targetVelocity=0, force=0) 

        Dd = np.eye(7) * 1000
        Kd = np.eye(7) * 1000

        joint_angles, joint_vels = self.getSpaceRobotDoFJointAnglesAndVelocityies()
        desired_acceleration = np.zeros(np.array(joint_angles).shape).tolist() # 0
        tau = np.array(p.calculateInverseDynamics(self.space_robot_id, joint_angles, joint_vels, desired_acceleration, flags=1))[self.arm_2_joint_DoF_id].reshape(-1, 1)

        current_angles = np.array(joint_angles)[self.arm_2_joint_DoF_id].reshape(-1, 1)
        current_vels = np.array(joint_vels)[self.arm_2_joint_DoF_id].reshape(-1, 1)
        desired_angles = desired_angles.reshape(-1, 1)

        if np.linalg.norm(desired_angles - current_angles) > 0.1:
            mid_desired_angles = current_angles + 0.1 * (desired_angles - current_angles)/np.linalg.norm(desired_angles - current_angles)
        else:
            mid_desired_angles = desired_angles

        tau += Dd @ (- current_vels) + Kd @ (mid_desired_angles - current_angles)
        tau = tau.reshape(-1, )

        max_joint_torque = 100
        tau = np.clip(tau, -max_joint_torque, max_joint_torque)
        for i, id in enumerate(self.arm_2_id):
            p.setJointMotorControl2(self.space_robot_id, id, p.TORQUE_CONTROL, force=tau[i] ) 

        # 保存数据
        self.joint_torque_in_moveToWorkPose.append(tau)

        if np.linalg.norm(desired_angles - current_angles) < 5e-2 and np.linalg.norm(current_vels) < 5e-2:
            current_angles = np.array(self.getSpaceRobotJointAngles())[self.arm_2_id]
            for i in range(0, len(self.arm_2_id)):
                p.setJointMotorControl2(self.space_robot_id, self.arm_2_id[i], p.POSITION_CONTROL, targetPosition=current_angles[i], positionGain=1, maxVelocity=100, force=100)
            return True

        return False


    # -----------------------------------------------------------------------------------------------
    def pickOutTool(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.space_robot_id)

        target_grasp_local_pos =  np.array(self.cutter_initial_local_pos) +  np.array([0, 1.5, 0])
        target_grasp_local_quat = p.getQuaternionFromEuler([np.pi/2, 0, 0])
        target_grasp_pos, target_grasp_quat = p.multiplyTransforms(base_pos, base_quat, 
                                                                target_grasp_local_pos, target_grasp_local_quat ) # np.pi/2, 0, np.pi

        end_link_id = self.arm_1_id[-1]
        joint_damping = 0.1 * np.ones((len(self.arm_1_id) + len(self.arm_1_id), ))
        joint_damping[self.arm_1_id[0]] = 10
        joint_damping[self.arm_1_id[1]] = 10
        joint_damping[self.arm_1_id[5]] = 10
        joint_angles = np.array(p.calculateInverseKinematics(self.space_robot_id, end_link_id, target_grasp_pos, targetOrientation=target_grasp_quat,
            jointDamping=joint_damping.tolist() ))[self.arm_1_joint_DoF_id]

        current_angles = np.array(self.getSpaceRobotJointAngles())[self.arm_1_id]
        target_angles = joint_angles
        scales = np.abs(target_angles - current_angles)
        if np.linalg.norm(scales) < 1e-2:
            return True

        max_vel = 0.5
        scales /= np.max(scales)
        for i, id in enumerate(self.arm_1_id):
            p.setJointMotorControl2(self.space_robot_id, id, p.POSITION_CONTROL, targetPosition=target_angles[i], positionGain=0.05, maxVelocity=max_vel * scales[i], force=1000) 
        return False


    # -----------------------------------------------------------------------------------------------
    def pickTool(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.space_robot_id)
        # 计算期望位姿
        target_grasp_local_pos =  np.array(self.cutter_initial_local_pos) +  np.array([0, 0.05, 0])
        target_grasp_local_quat = self.gripper_initial_local_quat
        target_grasp_pos, target_grasp_quat = p.multiplyTransforms(base_pos, base_quat, 
                                                                target_grasp_local_pos, target_grasp_local_quat ) # np.pi/2, 0, np.pi

        # 计算jacobian矩阵
        current_joint_angles, current_joint_vel = self.getSpaceRobotDoFJointAnglesAndVelocityies()
        linear_jacobian, angular_jacobian = p.calculateJacobian(self.space_robot_id, self.arm_1_id[-1], localPosition=[0,0,0], objPositions=current_joint_angles, 
                objVelocities=[0]*len(current_joint_angles), objAccelerations=[0]*len(current_joint_angles))

        whole_jacobian = np.concatenate([np.array(linear_jacobian), np.array(angular_jacobian)], axis=0)
        jacobian_in_base = whole_jacobian[:, 6 + np.array(self.arm_1_joint_DoF_id)] # 前6位是基座漂浮的6个自由度
        R_base_in_world = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)
        jacobian_in_world = block_diag(R_base_in_world, R_base_in_world) @ jacobian_in_base

        current_end_cm_pos = (p.getLinkState(self.space_robot_id, linkIndex=self.arm_1_id[-1])[0])
        current_end_cm_quat = (p.getLinkState(self.space_robot_id, linkIndex=self.arm_1_id[-1])[1])

        delta_end_pos = np.array(target_grasp_pos) - np.array(current_end_cm_pos)
        delta_end_quat = p.getDifferenceQuaternion(current_end_cm_quat, target_grasp_quat)
        axis, angle = p.getAxisAngleFromQuaternion(delta_end_quat)
        delta_end_ori = np.array(axis) * angle
        delta_end_pose = np.concatenate([delta_end_pos.reshape(-1, 1), delta_end_ori.reshape(-1, 1)], axis=0)

        # 计算关节角速度
        k = 1.0
        desired_end_vel = k * delta_end_pose
        joint_vel = k * np.linalg.pinv(jacobian_in_world) @ desired_end_vel
        joint_vel = joint_vel.reshape(-1, )
        # 控制关节角速度
        for i in range(0, len(self.arm_1_id)):
            p.setJointMotorControl2(self.space_robot_id, self.arm_1_id[i], p.VELOCITY_CONTROL, targetVelocity=joint_vel[i], force=100)

        # 若完成
        if np.linalg.norm(np.array(current_end_cm_pos) - np.array(target_grasp_pos)) < 1e-2 \
                and np.linalg.norm(np.array(current_end_cm_quat) - np.array(target_grasp_quat)) < 1e-2:

            # 清除base和cutter之间的fixed constraint
            p.removeConstraint(self.base_cutter_constraint_id)

            # 新建end 和 cutter之间的fixed constraint
            cutter_cm_pos, cutter_cm_quat = p.getBasePositionAndOrientation(self.cutter_id)
            current_end_cm_pos = p.getLinkState(self.space_robot_id, linkIndex=self.arm_1_id[-1])[0]
            current_end_cm_quat = p.getLinkState(self.space_robot_id, linkIndex=self.arm_1_id[-1])[1]

            joint_pos = current_end_cm_pos
            joint_quat = current_end_cm_quat
            current_end_cm_pos_inv, current_end_cm_quat_inv = p.invertTransform(current_end_cm_pos, current_end_cm_quat)
            joint_pos_in_parent, joint_quat_in_parent = p.multiplyTransforms(current_end_cm_pos_inv, current_end_cm_quat_inv, joint_pos, joint_quat)
            cutter_cm_pos_inv, cutter_cm_pos_quat_inv = p.invertTransform(cutter_cm_pos, cutter_cm_quat)
            joint_pos_in_child, joint_quat_in_child = p.multiplyTransforms(cutter_cm_pos_inv, cutter_cm_pos_quat_inv, joint_pos, joint_quat)
            
            self.end_cutter_constraint_id = p.createConstraint(parentBodyUniqueId=self.space_robot_id, parentLinkIndex=self.arm_1_id[-1], childBodyUniqueId=self.cutter_id, childLinkIndex=-1,
                jointType=p.JOINT_FIXED, jointAxis=[0, 1, 0], 
                parentFramePosition=joint_pos_in_parent, childFramePosition=joint_pos_in_child,
                parentFrameOrientation=joint_quat_in_parent, childFrameOrientation=joint_quat_in_child)
            return True

        return False


    # -----------------------------------------------------------------------------------------------
    def fixAntenna(self, stage):
        target_satellite_pos, target_satellite_quat = p.getBasePositionAndOrientation(self.target_satellite_id)

        if stage == 1:
            target_grasp_local_pos = [0, -1.79 - 1.6, 1.45]
        else:
            target_grasp_local_pos = [0, -1.79 - 1.0, 1.45]
        target_grasp_local_quat = p.getQuaternionFromEuler([np.pi/2, 0, np.pi])
        target_grasp_pos, target_grasp_quat = p.multiplyTransforms(target_satellite_pos, target_satellite_quat, 
                                                                target_grasp_local_pos, target_grasp_local_quat ) # np.pi/2, 0, np.pi

        end_link_id = self.arm_1_id[-1]
        joint_damping = 0.1 * np.ones((len(self.arm_1_id) + len(self.arm_1_id), ))
        joint_damping[self.arm_1_id[0]] = 10
        joint_damping[self.arm_1_id[1]] = 10
        joint_damping[self.arm_1_id[5]] = 10
        joint_angles = np.array(p.calculateInverseKinematics(self.space_robot_id, end_link_id, target_grasp_pos, targetOrientation=target_grasp_quat,
            jointDamping=joint_damping.tolist() ))[self.arm_1_joint_DoF_id]

        current_angles = np.array(self.getSpaceRobotJointAngles())[self.arm_1_id]
        target_angles = joint_angles
        scales = np.abs(target_angles - current_angles)
        if np.linalg.norm(scales) < 1e-2:
            return True

        max_vel = 0.2
        scales /= np.max(scales)
        for i, id in enumerate(self.arm_1_id):
            p.setJointMotorControl2(self.space_robot_id, id, p.POSITION_CONTROL, targetPosition=target_angles[i], positionGain=0.05, maxVelocity=max_vel * scales[i], force=1000) 
        return False

    
    # -----------------------------------------------------------------------------------------------
    def fixLeftPanel(self, stage):
        target_satellite_pos, target_satellite_quat = p.getBasePositionAndOrientation(self.target_satellite_id)

        if stage == 1:
            target_grasp_local_pos = [-1.5, -1.79 - 2.0, -1.0]
        else:
            target_grasp_local_pos = [-1.5, -1.79 - 0.9, -1.0]
        target_grasp_local_quat = p.getQuaternionFromEuler([np.pi/2, np.pi/2, np.pi])
        target_grasp_pos, target_grasp_quat = p.multiplyTransforms(target_satellite_pos, target_satellite_quat, 
                                                                target_grasp_local_pos, target_grasp_local_quat ) # np.pi/2, 0, np.pi

        end_link_id = self.arm_1_id[-1]
        joint_damping = 0.1 * np.ones((len(self.arm_1_id) + len(self.arm_1_id), ))
        joint_damping[self.arm_1_id[0]] = 10
        joint_damping[self.arm_1_id[1]] = 10
        joint_damping[self.arm_1_id[5]] = 10
        joint_angles = np.array(p.calculateInverseKinematics(self.space_robot_id, end_link_id, target_grasp_pos, targetOrientation=target_grasp_quat,
            jointDamping=joint_damping.tolist() ))[self.arm_1_joint_DoF_id]

        current_angles = np.array(self.getSpaceRobotJointAngles())[self.arm_1_id]
        target_angles = joint_angles
        scales = np.abs(target_angles - current_angles)
        if np.linalg.norm(scales) < 1e-2:
            return True

        max_vel = 0.2
        scales /= np.max(scales)
        for i, id in enumerate(self.arm_1_id):
            p.setJointMotorControl2(self.space_robot_id, id, p.POSITION_CONTROL, targetPosition=target_angles[i], positionGain=0.05, maxVelocity=max_vel * scales[i], force=1000) 
        return False


    # -----------------------------------------------------------------------------------------------
    def fixRightPanel(self, stage):
        target_satellite_pos, target_satellite_quat = p.getBasePositionAndOrientation(self.target_satellite_id)

        if stage == 1:
            target_grasp_local_pos = [1.5, -1.79 - 2.0, -1.0]
        else:
            target_grasp_local_pos = [1.5, -1.79 - 0.9, -1.0]
        target_grasp_local_quat = p.getQuaternionFromEuler([np.pi/2, np.pi/2, np.pi])
        target_grasp_pos, target_grasp_quat = p.multiplyTransforms(target_satellite_pos, target_satellite_quat, 
                                                                target_grasp_local_pos, target_grasp_local_quat ) # np.pi/2, 0, np.pi

        end_link_id = self.arm_1_id[-1]
        joint_damping = 0.1 * np.ones((len(self.arm_1_id) + len(self.arm_1_id), ))
        joint_damping[self.arm_1_id[0]] = 10
        joint_damping[self.arm_1_id[1]] = 10
        joint_damping[self.arm_1_id[5]] = 10
        joint_angles = np.array(p.calculateInverseKinematics(self.space_robot_id, end_link_id, target_grasp_pos, targetOrientation=target_grasp_quat,
            jointDamping=joint_damping.tolist() ))[self.arm_1_joint_DoF_id]

        current_angles = np.array(self.getSpaceRobotJointAngles())[self.arm_1_id]
        target_angles = joint_angles
        scales = np.abs(target_angles - current_angles)
        if np.linalg.norm(scales) < 1e-2:
            return True

        max_vel = 0.2
        scales /= np.max(scales)
        for i, id in enumerate(self.arm_1_id):
            p.setJointMotorControl2(self.space_robot_id, id, p.POSITION_CONTROL, targetPosition=target_angles[i], positionGain=0.05, maxVelocity=max_vel * scales[i], force=1000) 
        return False


    # -----------------------------------------------------------------------------------------------
    def baseControl(self, desired_pos=None, desired_lvel=None, desired_quat=None, desired_avel=None):
        flags = p.WORLD_FRAME
        base_pos, base_quat = p.getBasePositionAndOrientation(self.space_robot_id)
        base_lvel, base_avel = p.getBaseVelocity(self.space_robot_id)

        delta_avel = np.array(desired_avel) - np.array(base_avel)
        if desired_quat == None:
            kv = 10000
            torque =  kv * delta_avel
        else:
            delta_quat = p.getDifferenceQuaternion(base_quat, desired_quat)
            axis, angle = p.getAxisAngleFromQuaternion(delta_quat)
            delta_ori = np.array(axis) * angle
            kp = 10000
            kv = 10000
            torque = kp * delta_ori + kv * delta_avel
        
        torque = np.clip(torque, -100, 100)
        p.applyExternalTorque(self.space_robot_id, -1, torqueObj=torque, flags=flags)


        delta_lvel = np.array(desired_lvel) - np.array(base_lvel)
        if desired_pos == None:
            kv = 1000
            force =  kv * delta_lvel
        else:
            delta_pos = desired_pos - base_pos
            kp = 1000
            kv = 10000
            force = kp * delta_pos + kv * delta_lvel
        
        force = np.clip(force, -100, 100)
        p.applyExternalForce(self.space_robot_id, -1, forceObj=force, posObj=base_pos, flags=flags)

        return force, torque


    # -----------------------------------------------------------------------------------------------
    def returnAndBack(self):
        self.baseControl(desired_pos=None, desired_lvel=[0, -0.1, 0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0, 0, 0])

        max_vel = 0.5
        
        current_angles = np.array(self.getSpaceRobotJointAngles())[self.arm_1_id]
        target_angles = np.array([1/2*np.pi, 0, 1/2*np.pi, -1*np.pi, 1*np.pi, 0, 0])
        scales = np.abs(target_angles - current_angles)

        scales /= np.max(scales)
        for i, id in enumerate(self.arm_1_id):
            p.setJointMotorControl2(self.space_robot_id, id, p.POSITION_CONTROL, targetPosition=target_angles[i], positionGain=0.1, maxVelocity=max_vel * scales[i], force=1000) 

        current_angles = np.array(self.getSpaceRobotJointAngles())[self.arm_2_id]
        target_angles = np.array([-1/2*np.pi, 0, 1/2*np.pi, -1*np.pi, 1*np.pi, 0, 0])
        scales = np.abs(target_angles - current_angles)

        scales /= np.max(scales)
        for i, id in enumerate(self.arm_2_id):
            p.setJointMotorControl2(self.space_robot_id, id, p.POSITION_CONTROL, targetPosition=target_angles[i], positionGain=0.1, maxVelocity=max_vel * scales[i], force=1000) 



    # -----------------------------------------------------------------------------------------------
    def main(self):
        self.initiateScene()
        p.stepSimulation()


        # time.sleep(2)
        t_start = time.time()

        # 仿真loop
        stage = 0
        count = 0
        while True:

            # if self.sim_count % 24 == 0:
            #     self.camera1()
            #     self.camera2()

            if stage < 8:
                p.resetBaseVelocity(self.target_satellite_id, linearVelocity=[-0.01, 0.01, 0.01], angularVelocity=[0.03, -0.157, 0.03])

             # ---------------------------------------------------------------------------------------------------------------
            if stage == 0:
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                # p.resetBaseVelocity(self.space_robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
                if self.arm2ToInitialWorkConfiguration(max_vel=0.5):
                    stage += 1

            elif stage == 1:
                # p.resetBaseVelocity(self.space_robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                if self.pickOutGripper():
                    stage += 1

            elif stage == 2:
                # p.resetBaseVelocity(self.space_robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                if self.pickGripper():
                    stage += 1

            elif stage == 3:
                # p.resetBaseVelocity(self.space_robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                if self.pickOutGripper():
                    stage += 1

            elif stage == 4:
                # p.resetBaseVelocity(self.space_robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                if self.arm2ToInitialWorkConfiguration(max_vel=0.5):
                    stage += 1

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 5:
                if self.baseMove():
                    stage += 1
                    

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 6:
                self.baseMove()
                if self.moveToPreGraspPose2():
                    stage += 1

             # --------------------------------------------------------------------------------------------------------------
            elif stage == 7:
                if self.makeGraspConstraint():
                    stage += 1

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 8:
                if self.stopMotion():
                    stage += 1

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 9:
                if self.recoverBaseControlled():
                    stage += 1

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 10:
                force, torque = self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                self.base_force_in_moveToWorkPose.append(np.concatenate([force, torque]))

                if self.moveTargetToWorkPose():
                    stage += 1

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 11:
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                if self.arm1ToInitialWorkConfiguration(max_vel=0.5):
                    stage += 1

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 12:
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                if self.pickOutTool():
                    stage += 1

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 13:
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                if self.pickTool():
                    stage += 1
                    pass

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 14:
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                if self.pickOutTool():
                    stage += 1

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 15:
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                if self.arm1ToInitialWorkConfiguration():
                    stage += 1

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 16:
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                if self.fixAntenna(stage=1):
                    stage += 1

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 17:
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                if self.fixAntenna(stage=2):
                    stage += 1

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 18:
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                self.openTargetSatellite('antenna_2')
                if self.fixAntenna(stage=1):
                    stage += 1

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 19:
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                if self.fixLeftPanel(stage=1):
                    stage += 1

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 20:
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                if self.fixLeftPanel(stage=2):
                    stage += 1

            
            # --------------------------------------------------------------------------------------------------------------
            elif stage == 21:
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                self.openTargetSatellite('panel_1-4')
                if self.fixLeftPanel(stage=1):
                    stage += 1

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 22:
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                if self.fixRightPanel(stage=1):
                    stage += 1

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 23:
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                if self.fixRightPanel(stage=2):
                    stage += 1

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 24:
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                self.openTargetSatellite('panel_5-8')
                if self.fixRightPanel(stage=1):
                    stage += 1

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 25:
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])
                if self.arm1ToInitialWorkConfiguration():
                    stage += 1
                    p.removeConstraint(self.grasp_constraint_id)
                    count = self.sim_count

            # --------------------------------------------------------------------------------------------------------------
            elif stage == 26:
                p.resetBaseVelocity(self.target_satellite_id, linearVelocity=[0,0,0], angularVelocity=[0,0,0])
                self.returnAndBack()
                if (self.sim_count - count) == 240 * 30:
                    break

            # # --------------------------------------------------------------------------------------------------------------
            else:
                self.baseControl(desired_pos=None, desired_lvel=[0,0,0], desired_quat=p.getQuaternionFromEuler([0,0,0]), desired_avel=[0,0,0])

            # print(stage)

            p.stepSimulation()
            # time.sleep(1.0 / self.sim_rate)
            self.sim_count += 1
            # print((time.time() - t_start) / self.sim_count * self.sim_rate)


        self.video1.release()
        self.video2.release()

        cv2.destroyAllWindows()
        p.disconnect()

        np.save(self.project_dir + "results/base_force_in_baseMove.npy", self.base_force_in_baseMove)
        np.save(self.project_dir + "results/base_pos_in_baseMove.npy", self.base_pos_in_baseMove)
        np.save(self.project_dir + "results/target_pos_in_baseMove.npy", self.target_pos_in_baseMove)
        np.save(self.project_dir + "results/base_avel_in_baseMove.npy", self.base_avel_in_baseMove)
        np.save(self.project_dir + "results/target_avel_in_baseMove.npy", self.target_avel_in_baseMove)

        np.save(self.project_dir + "results/end_pos_in_moveToGraspPoint.npy", self.end_pos_in_moveToGraspPoint)
        np.save(self.project_dir + "results/target_pos_in_moveToGraspPoint.npy", self.target_pos_in_moveToGraspPoint)

        np.save(self.project_dir + "results/base_torque_in_stopMotion.npy", self.base_torque_in_stopMotion)
        np.save(self.project_dir + "results/end_6DoF_force_in_stopMotion.npy", self.end_6DoF_force_in_stopMotion)
        np.save(self.project_dir + "results/joint_torque_in_stopMotion.npy", self.joint_torque_in_stopMotion)
        np.save(self.project_dir + "results/joint_angle_in_stopMotion.npy", self.joint_angle_in_stopMotion)

        np.save(self.project_dir + "results/base_force_in_moveToWorkPose.npy", self.base_force_in_moveToWorkPose)
        np.save(self.project_dir + "results/joint_torque_in_moveToWorkPose.npy", self.joint_torque_in_moveToWorkPose)



# ---------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    env = Environment()
    env.main()