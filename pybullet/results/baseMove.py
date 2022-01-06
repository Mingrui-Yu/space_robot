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
def base_force_in_baseMove():
    # end_force_in_stopMotion
    base_force = np.load(project_dir + "results/base_force_in_baseMove.npy")

    time = np.arange(base_force.shape[0]) / 240

    plt.figure(figsize=(14,7))

    plt.subplot(1, 2, 1)
    plt.plot(time, base_force[:, 0], linewidth=3, label="X axis")
    plt.plot(time, base_force[:, 1], linewidth=3, label="Y axis")
    plt.plot(time, base_force[:, 2], linewidth=3, label="Z axis")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Applied force on base (N)')
    plt.xlim([0, time[-1]])

    plt.subplot(1, 2, 2)
    plt.plot(time, base_force[:, 3], linewidth=3, label="X axis")
    plt.plot(time, base_force[:, 4], linewidth=3, label="Y axis")
    plt.plot(time, base_force[:, 5], linewidth=3, label="Z axis")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Applied torque on base (N $\cdot$ m)')
    plt.xlim([0, time[-1]])

    plt.subplots_adjust(left=0.1, bottom=0.12, right=0.95, top=0.95, wspace=0.3)






# -----------------------------------------------------------------------------
def base_avel_in_baseMove():
    base_avel = np.load(project_dir + "results/base_avel_in_baseMove.npy")
    target_avel = np.load(project_dir + "results/target_avel_in_baseMove.npy")

    time = np.arange(base_avel.shape[0]) / 240

    plt.figure(figsize=(10,7))
    plt.plot(time, base_avel[:, 1], linewidth=3, label="base")
    plt.plot(time, target_avel[:, 1], linewidth=3, label="target satellite")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Angular velocity on Y axis (rad/s)')
    plt.xlim([0, time[-1]])




# -----------------------------------------------------------------------------
if __name__ == '__main__':

    base_force_in_baseMove()

    base_avel_in_baseMove()

    plt.show()