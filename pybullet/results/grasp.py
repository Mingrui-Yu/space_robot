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
def base_pos():
    end_pos = np.load(project_dir + "results/end_pos_in_moveToGraspPoint.npy")
    target_pos = np.load(project_dir + "results/target_pos_in_moveToGraspPoint.npy")

    time = np.arange(end_pos.shape[0]) / 240

    plt.figure(figsize=(14,5))

    plt.subplot(1, 3, 1)
    plt.plot(time, end_pos[:, 0], linewidth=3, label="Current end")
    plt.plot(time, target_pos[:, 0], linewidth=3, label="target satellite")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.xlim([0, time[-1]])
    plt.title('X axis')

    plt.subplot(1, 3, 2)
    plt.plot(time, end_pos[:, 1], linewidth=3, label="Current end")
    plt.plot(time, target_pos[:, 1], linewidth=3, label="Target")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.xlim([0, time[-1]])
    plt.title('Y axis')

    plt.subplot(1, 3, 3)
    plt.plot(time, end_pos[:, 2], linewidth=3, label="Current end")
    plt.plot(time, target_pos[:, 2], linewidth=3, label="Target")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.xlim([0, time[-1]])
    plt.title('Z axis')

    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.92, wspace=0.4)




# -----------------------------------------------------------------------------
if __name__ == '__main__':

    base_pos()

    plt.show()