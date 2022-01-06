import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties 

params={#'font.family':'serif',
#                     'font.serif':'Times New Roman',
#                     # 'font.style':'italic',
#                     'font.weight':'normal', #or 'bold'
                    'font.size': 20, #or large,small
                    'pdf.fonttype': 42,
                    'ps.fonttype': 42
                    }
rcParams.update(params)


project_dir = '/home/mingrui/Mingrui/Homework/space_robot/'



# -----------------------------------------------------------------------------
def base_torque_in_stopMotion():
    # base_torque_in_stopMotion
    base_torque = np.load(project_dir + "results/base_torque_in_stopMotion.npy")
    time = np.arange(base_torque.shape[0]) / 240

    plt.figure(figsize=(10,7))
    plt.plot(time, base_torque[:, 0], linewidth=3, label="X axis")
    plt.plot(time, base_torque[:, 1], linewidth=3, label="Y axis")
    plt.plot(time, base_torque[:, 2], linewidth=3, label="Z axis")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Applied torque on base (N $\cdot$ m)')
    plt.xlim([0, time[-1]])



# -----------------------------------------------------------------------------
def end_6DoF_force_in_stopMotion():
    # end_force_in_stopMotion
    end_force = np.load(project_dir + "results/end_6DoF_force_in_stopMotion.npy")
    end_force = end_force[24:, :]

    time = np.arange(end_force.shape[0]) / 240

    plt.figure(figsize=(14,7))

    plt.subplot(1, 2, 1)
    plt.plot(time, end_force[:, 0], linewidth=3, label="X axis")
    plt.plot(time, end_force[:, 1], linewidth=3, label="Y axis")
    plt.plot(time, end_force[:, 2], linewidth=3, label="Z axis")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Force on grasp point (N)')
    plt.xlim([0, time[-1]])

    plt.subplot(1, 2, 2)
    plt.plot(time, end_force[:, 3], linewidth=3, label="X axis")
    plt.plot(time, end_force[:, 4], linewidth=3, label="Y axis")
    plt.plot(time, end_force[:, 5], linewidth=3, label="Z axis")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Torque on grasp point (N $\cdot$ m)')
    plt.xlim([0, time[-1]])

    plt.subplots_adjust(left=0.1, bottom=0.12, right=0.95, top=0.95, wspace=0.3)




# -----------------------------------------------------------------------------
def joint_torque_in_stopMotion():
    joint_torque = np.load(project_dir + "results/joint_torque_in_stopMotion.npy")
    time = np.arange(joint_torque.shape[0]) / 240

    plt.figure(figsize=(10,7))
    plt.plot(time, joint_torque[:, 0], linewidth=3, label="Joint 1")
    plt.plot(time, joint_torque[:, 1], linewidth=3, label="Joint 2")
    plt.plot(time, joint_torque[:, 2], linewidth=3, label="Joint 3")
    plt.plot(time, joint_torque[:, 3], linewidth=3, label="Joint 4")
    plt.plot(time, joint_torque[:, 4], linewidth=3, label="Joint 5")
    plt.plot(time, joint_torque[:, 5], linewidth=3, label="Joint 6")
    plt.plot(time, joint_torque[:, 6], linewidth=3, label="Joint 7")

    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Applied torque on joints (N $\cdot$ m)')
    plt.xlim([0, time[-1]])

    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.95)



# -----------------------------------------------------------------------------
def joint_angle_in_stopMotion():
    joint_angle = np.load(project_dir + "results/joint_angle_in_stopMotion.npy")
    time = np.arange(joint_angle.shape[0]) / 240

    plt.figure(figsize=(10,7))
    plt.plot(time, joint_angle[:, 0], linewidth=3, label="Joint 1")
    plt.plot(time, joint_angle[:, 1], linewidth=3, label="Joint 2")
    plt.plot(time, joint_angle[:, 2], linewidth=3, label="Joint 3")
    plt.plot(time, joint_angle[:, 3], linewidth=3, label="Joint 4")
    plt.plot(time, joint_angle[:, 4], linewidth=3, label="Joint 5")
    plt.plot(time, joint_angle[:, 5], linewidth=3, label="Joint 6")
    plt.plot(time, joint_angle[:, 6], linewidth=3, label="Joint 7")

    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint angles (rad)')
    plt.xlim([0, time[-1]])

    plt.subplots_adjust(left=0.15, bottom=0.12, right=0.95, top=0.95)



# -----------------------------------------------------------------------------
if __name__ == '__main__':

    base_torque_in_stopMotion()

    end_6DoF_force_in_stopMotion()

    joint_torque_in_stopMotion()

    joint_angle_in_stopMotion()

    plt.show()