## 绘画Polygon、line、polylines

```python

import cv2
import numpy as np
from PIL import Image, ImageDraw

# 图像路径
image_path = "rs19_val/jpgs/rs19_val/rs00000.jpg"

# 分割标注数据
left_rail = [[901, 1079], [902, 1070], [903, 1060], [904, 1051], [905, 1041], [906, 1032], [908, 1022], [908, 1013], [910, 1003], [911, 993], [912, 984], [914, 974], [914, 964], [916, 955], [917, 946], [918, 936], [919, 926], [921, 917], [921, 907], [923, 898], [924, 888], [925, 879], [926, 869], [927, 859], [928, 850], [930, 841], [930, 831], [932, 821], [933, 812], [934, 802], [935, 792], [936, 783], [937, 773], [939, 764], [940, 754], [942, 745], [944, 736], [946, 727], [948, 718], [951, 709], [953, 700], [956, 691], [958, 682], [961, 673], [965, 664], [968, 656], [972, 648], [977, 640], [983, 632], [989, 625], [997, 618], [1004, 613], [1013, 608], [1022, 604], [1031, 600], [1040, 597], [1052, 594], [1062, 592]]
right_rail = [[1166, 1079], [1164, 1071], [1160, 1062], [1156, 1054], [1153, 1045], [1150, 1037], [1146, 1028], [1142, 1020], [1139, 1011], [1135, 1003], [1132, 994], [1129, 986], [1125, 978], [1122, 969], [1118, 960], [1114, 952], [1111, 943], [1107, 935], [1104, 926], [1100, 918], [1097, 909], [1094, 901], [1090, 892], [1086, 884], [1083, 875], [1080, 866], [1077, 858], [1073, 849], [1070, 841], [1067, 832], [1063, 824], [1060, 815], [1057, 807], [1053, 798], [1050, 789], [1047, 781], [1043, 772], [1040, 764], [1036, 755], [1034, 746], [1032, 738], [1029, 728], [1027, 720], [1024, 711], [1022, 701], [1019, 693], [1016, 684], [1014, 675], [1013, 665], [1012, 656], [1013, 646], [1014, 637], [1016, 628], [1020, 620], [1026, 613], [1034, 607], [1043, 603], [1051, 600], [1063, 596]]

# 将标注点转换为numpy数组
left_rail = np.array(left_rail, dtype=np.int32)
right_rail = np.array(right_rail, dtype=np.int32)

# 读取图像
image = cv2.imread(image_path)

# 复制图像用于绘制
visualized_image = image.copy()

# 绘制左铁轨
for i in range(len(left_rail) - 1):
    cv2.line(visualized_image, tuple(left_rail[i]), tuple(left_rail[i + 1]), (0, 255, 0), 2)

# 绘制右铁轨
for i in range(len(right_rail) - 1):
    cv2.line(visualized_image, tuple(right_rail[i]), tuple(right_rail[i + 1]), (0, 0, 255), 2)

# 显示结果
# cv2.imshow("Rail Track Visualization", visualized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 保存可视化结果
cv2.imwrite("rail_track_visualization.jpg", visualized_image)

"""
# 创建空白图像用于绘制区域  
overlay = np.zeros_like(image)  
  
# 构建闭合区域的多边形顶点  
# 将左铁轨和右铁轨的点合并，并添加首尾连接点  
polygon_points = np.vstack((left_rail, right_rail[::-1] ))   # left_rail[0].reshape(1,2), right_rail[-1].reshape(1,2)
  
# 绘制填充区域（使用半透明蓝色）  
cv2.fillPoly(overlay, [polygon_points], (255, 0, 0), cv2.LINE_AA)  
  
# 将覆盖层与原图叠加  
result_image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)  
  
# 保存结果图像  
cv2.imwrite('rail_area_visualization.jpg', result_image)  
"""

"""
points = np.array(points, dtype=np.int32)

img = cv2.imread("frame.jpg")
for i in range(len(points)-1):
    pt1 = points[i]
    pt2 = points[i+1]
    # 划线
    # cv2.line(img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 255, 0), 2)

# 多边形线条
# cv2.polylines(img, [points], isClosed=False, color=(255, 0, 0), thickness=3)
cv2.fillPoly(img, [points], (255, 0, 0))
cv2.imwrite("frame_visualization.jpg", img)
"""

```



```python
def generate_rails_mask(shape):
    rails_mask = Image.new("L", shape, 0)
    draw = ImageDraw.Draw(rails_mask)

    rails = [np.array(left_rail), np.array(right_rail)]
    for rail in rails:
        draw.line([tuple(xy) for xy in rail], fill=1, width=10) # 绘制白色（值1）线条，线宽10
    rails_mask.save('after_draw_line.png')
    rails_mask = np.array(rails_mask)
    rails_mask[: max(rails[0][:, 1].min(), rails[1][:, 1].min()), :] = 0 # 清除轨道顶部以上的无效区域。
    for row_idx in np.where(np.sum(rails_mask, axis=1) > 2)[0]: # 遍历非零像素数>2的行
        rails_mask[row_idx, np.nonzero(rails_mask[row_idx, :])[0][1:-1]] = 0 # # 获取该行所有非零像素列索引,# 保留首尾像素，清除中间像素
    return rails_mask

shape = image.shape[:2][::-1]
output = generate_rails_mask(shape)

draw.line([0, 0, 100, 100], fill='red', width=2) #画一条红色直线。‌

draw.rectangle([60, 60, 120, 80], fill='blue', outline='white') # 画一个蓝色填充的矩形。‌

draw.polygon(xy, fill=None, outline=None)  # 绘制多边形

draw.text((10,10), "Hello", fill='black') # 添加文字

```
