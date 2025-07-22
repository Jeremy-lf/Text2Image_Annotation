from matplotlib.font_manager import font_scalings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#下面两句代码防止中文显示成方块
# from pylab import mpl
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# """

plt.rcParams['figure.dpi'] = 500 #图片像素
plt.rcParams['figure.figsize'] = (3.71, 3.71)

# x_bits = [12, 24, 32, 48]
def joint_early():
    with PdfPages('latency.pdf') as pdf:
        # 开始绘制
        colors = ['red', 'orange', 'blue', 'green', 'black', 'purple','brown']
        markers = ['o', 'v', '*', 's', '+', 'd', '^']


       

        x_2 = [ 3.42, 5.63, 9.02]
        y_2 = [44.3, 49.1, 51.8]
        plt.plot(x_2, y_2, color=colors[2], 
                     marker=markers[2], linestyle='--', label='YOLOV6-3.0', markersize=3)


    
        x_1 = [7.07, 9.5, 12.39, 16.86]
        y_1 = [44.9, 50.6, 52.9, 53.9]
        plt.plot(x_1, y_1, color=colors[1], 
                     marker=markers[1], linestyle='--', label='YOLOv8', markersize=3)

        
        x_1 = [10.57]#, 12.5, 15.0, 17.5, 20.0]
        y_1 = [52.5]
        plt.plot(x_1, y_1, color=colors[4], 
                     marker=markers[4], linestyle='--', label='YOLOv9', markersize=3)


        x_7 = [3.82, 6.39, 10.65]
        y_7 = [45.4, 49.8, 51.8]
        plt.plot(x_7, y_7, color=colors[6], 
                     marker=markers[6], linestyle='--', label='Gold-YOLO', markersize=3)


        x_6 = [4.8, 7.85, 10.6, 17.05]
        y_6 = [43.1, 48.9, 51.4, 52.2]
        plt.plot(x_6, y_6, color=colors[5], 
                     marker=markers[5], linestyle='--', label='PP-YOLOE', markersize=3)

        
        x_0 = [4.58,  6.90, 9.20, 13.71]#, 12.5, 15.0, 17.5, 20.0]
        y_0 = [46.5,  51.3, 53.1, 54.3]
        plt.plot(x_0, y_0, color=colors[3], 
                     marker=markers[0], linestyle='--', label='RT-DETR', markersize=3)

        
        x_3 = [4.58, 6.89, 9.20, 13.71]#, 12.5, 15.0, 17.5, 20.0]
        y_3 = [48.7, 51.8, 53.5, 54.6]
        plt.plot(x_3, y_3, color=colors[0], 
                     marker=markers[3], linestyle='-', label='RT-DETRv3(ours)', markersize=3)
        
        texts = ['R18', 'R50', 'R50','R101']
        dxy = [(1.05, 0.4), (1.5, 0.55), (0.65, 0.4), (0.65, 0.4) ]
        plt.legend(fontsize=8)
        # 一共有多少个点就循环多少次
        for i in range(len(x_3)):
            plt.text(x_3[i] - dxy[i][0], y_3[i] + dxy[i][1], texts[i], color=colors[0], fontsize=6)

        font = {'family': 'TimesNewRoman.ttf'}

        # plt.rcParams['font.family'] = ['serif']
        # plt.rcParams['font.sans-serif'] = ['Times New Roman']
        x_bits = [4, 8,  12,  16, 20]
        plt.xticks(x_bits)  # 横轴只有这四个刻度
        # plt.yticks(y_bits)
        plt.ylim(42.0, 56.0)      #y坐标范围
        plt.title("MS COCO Object Detection", fontsize=10)
        plt.xlabel("Latency T4 TensorRT FP16(ms)")  # 作用为横坐标轴添加标签  fontsize=12
        plt.ylabel("COCO AP(%)")  # 作用为纵坐标轴添加标签
        plt.legend(fontsize=8, prop=font)

        # plt.rcParams['font.weight'] = 'bold'


        plt.text(6.89-1.85, 51.8, 'Scaled', color=colors[0],fontsize=6)
        # plt.show()
        pdf.savefig(bbox_inches='tight')  # 如果要保存，就需要去掉plt.show()，因为plt.show()会把figure清除掉


joint_early()





"""
plt.rcParams['figure.dpi'] = 500 #图片像素
plt.rcParams['figure.figsize'] = (3.71, 3.71)

# x_bits = [12, 24, 32, 48]
def joint_early():
    with PdfPages('shoulian.pdf') as pdf:
        # 开始绘制
        colors = ['red', 'orange', 'blue', 'green', 'black', 'purple','brown']
        markers = ['o', 'v', '*', 's', '+', 'd', '^']


       
        # RT-DETR-R18
        x_2 = [12, 24,  36, 48, 60, 72]
        y_2 = [38.7, 42.6, 44.5, 45.6, 46.1, 46.5]
        y_21 = [42.8, 46.0, 47.5, 48.1, 48.5, 48.9]
        plt.plot(x_2, y_2, color=colors[3], 
                     marker=markers[3], linestyle='--', label='RT-DETR-R18', markersize=3)

        # plt.plot(x_2, y_21, color=colors[2], 
        #              marker=markers[2], linestyle='--', label='RT-DETR-R34', markersize=3)

        # RT-DETRv2-R18
        x_1 = [12, 24,  36, 48, 60, 72]
        y_1 = [39.8, 43.3, 44.9, 45.9,46.4, 46.7]
        y_11 = [43.0, 45.8, 47.2, 48.1, 48.7, 49.0]
        plt.plot(x_1, y_1, color=colors[1], 
                     marker=markers[1], linestyle='--', label='RT-DETRv2-R18', markersize=3)

        # plt.plot(x_1, y_11, color=colors[1], 
        #              marker=markers[1], linestyle='--', label='RT-DETRv2-R34', markersize=3)

        
       # RT-DETRv2-R18
        x_3 = [12, 24,  36, 48, 60, 72]
        y_3 = [41.5, 44.4, 46.1, 47.1,47.6, 48.1]
        y_33 = [44.7, 47.3, 48.6, 49.4,49.7, 49.9]

        plt.plot(x_3, y_3, color=colors[0], 
                     marker=markers[5], linestyle='-', label='RT-DETRv3-R18(ours)', markersize=3)
        # plt.plot(x_3, y_33, color=colors[3], 
        #              marker=markers[3], linestyle='-', label='RT-DETRv3-R34', markersize=3)
        # texts = ['R18', 'R50_m', 'R50','R101'] 
        # 一共有多少个点就循环多少次
        # for i in range(len(x_3)):
        #     plt.text(x_3[i]-0.7, y_3[i]+0.7, texts[i], color=colors[0])


        # plt.arrow(36, 46.1, 72-36, 46.7-46.1, head_width=0.5, head_length=0.5, fc='blue', ec='purple', linestyle='--')
        x_bits = [12, 24,  36, 48, 60, 72]
        plt.xticks(x_bits)  # 横轴只有这四个刻度
        # y_bits = ['1x', '2x', '3x', '4x', '5x', '6x']
        # plt.yticks(y_bits)
        plt.ylim(38.0, 49)      #y坐标范围
        plt.title("MS COCO Object Detection")
        plt.xlabel("Training Schedule(epochs)")  # 作用为横坐标轴添加标签  fontsize=12
        plt.ylabel("COCO AP(%)")  # 作用为纵坐标轴添加标签
        plt.legend()
        plt.grid(True)
        # plt.show()
        pdf.savefig()  # 如果要保存，就需要去掉plt.show()，因为plt.show()会把figure清除掉


joint_early()
"""


"""
plt.rcParams['figure.dpi'] = 500 #图片像素
plt.rcParams['figure.figsize'] = (3.71, 3.71)

def joint_early():
    with PdfPages('shoulian.pdf') as pdf:
        # 开始绘制
        colors = ['red', 'orange', 'blue', 'green', 'black', 'purple','brown']
        markers = ['o', 'v', '*', 's', '+', 'd', '^']


       
        # RT-DETR-R18
        x_2 = [12, 24,  36, 48, 60, 72]
        y_2 = [38.7, 42.6, 44.5, 45.6, 46.1, 46.5]
        y_21 = [42.8, 46.0, 47.5, 48.1, 48.5, 48.9]
        # plt.plot(x_2, y_2, 'o', color=colors[2], markersize=3)
        plt.plot(x_2, y_2, color=colors[2], 
                     marker=markers[2], linestyle='--', label='RT-DETR-R18', markersize=3)

        # plt.plot(x_2, y_21, color=colors[2], 
        #              marker=markers[2], linestyle='--', label='RT-DETR-R34', markersize=3)


        # RT-DETRv2-R18
        x_1 = [12, 24,  36, 48, 60, 72]
        y_1 = [39.8, 43.3, 44.9, 45.9,46.4, 46.7]
        y_11 = [43.0, 45.8, 47.2, 48.1, 48.7, 49.0]
        # plt.plot(x_1, y_1, 'o', color=colors[1], label='RT-DETRv2-R18', markersize=3)
        plt.plot(x_1, y_1, color=colors[1], 
                     marker=markers[1], linestyle='--', label='RT-DETRv2-R18', markersize=3)

        # plt.plot(x_1, y_11, color=colors[1], 
        #              marker=markers[1], linestyle='--', label='RT-DETRv2-R34', markersize=3)

        
       # RT-DETRv2-R18
        x_3 = [12, 24,  36, 48, 60, 72]
        y_3 = [41.5, 44.4, 46.1, 47.1,47.6, 48.1]
        y_33 = [44.7, 47.3, 48.6, 49.4,49.7, 49.9]

        plt.plot(x_3, y_3, color=colors[0], 
                     marker=markers[5], linestyle='-', label='RT-DETRv3-R18(ours)', markersize=3)
        
         # width=0.2,head_width=0.5,head_length=0.2
        # plt.arrow(36, 46.1, 72-36, 46.7-46.1, head_width=0.5, head_length=1, fc='purple', ec='purple', linestyle='--')
        # plt.arrow(72, 46.7, 45.5-72, 46.6-46.7, width=0.001, head_width=0.25, head_length=0.8, fc='orange', ec='orange', linestyle='-')
        
        # plt.arrow(44, 46.7, 72-46, 46.7-46.65, width=0.0005, head_width=0.25, head_length=0.8, fc='black', ec='black', linestyle='--')
        
        plt.arrow(44, 46.75, 72-46, 46.75-46.75, width=0.0001, head_width=0.25, head_length=0.9, fc='black', ec='black', linestyle='-')

        plt.text(51.9, 46.9, 'reduce 36% epochs', color='black', fontsize=5.6)
        plt.plot(44, 46.75, '*', color='black')
        # plt.arrow(60, 46.1, 36-56.5, 46.1-46.1, head_width=0.35, head_length=1, fc='blue', ec='blue', linestyle='--')

        # plt.plot(x_3, y_33, color=colors[3], 
        #              marker=markers[3], linestyle='-', label='RT-DETRv3-R34', markersize=3)
        # texts = ['R18', 'R50_m', 'R50','R101'] 
        # 一共有多少个点就循环多少次
        # for i in range(len(x_3)):
        #     plt.text(x_3[i]-0.7, y_3[i]+0.7, texts[i], color=colors[0])


        # plt.arrow(36, 46.1, 72-36, 46.7-46.1, head_width=0.5, head_length=0.5, fc='blue', ec='purple', linestyle='--')
        x_bits = [12, 24,  36, 48, 60, 72]
        plt.xticks(x_bits)  # 横轴只有这四个刻度
        # y_bits = ['1x', '2x', '3x', '4x', '5x', '6x']
        # plt.yticks(y_bits)
        plt.ylim(38.0, 49)      #y坐标范围
        plt.title("MS COCO Object Detection", fontsize=10)
        plt.xlabel("Training Schedule(epochs)")  # 作用为横坐标轴添加标签  fontsize=12
        plt.ylabel("COCO AP(%)")  # 作用为纵坐标轴添加标签
        plt.legend(fontsize=8)
        # plt.grid(True)
        # plt.show()
        pdf.savefig(bbox_inches='tight')  # 如果要保存，就需要去掉plt.show()，因为plt.show()会把figure清除掉


joint_early()
"""


