import numpy as np
from scipy.spatial.transform import Rotation as R
#('y', -np.pi/2, degrees=False)
rot1 = np.eye(4)
rot1[:3,:3] = R.from_euler('y', np.pi/2, degrees=False).as_matrix()
rot2 = np.eye(4)
rot2[:3,:3] = R.from_euler('x', np.pi/2, degrees=False).as_matrix()
def convert_coordinate_system(coord, conversion_type='left_to_right', 
                              angle_unit='degrees', 
                              apply_extra_rotation=True):
    """
    坐标系转换函数：支持位置与欧拉角旋转的转换，并可选应用额外旋转

    参数:
        coord: 输入数组 [x, y, z, rx, ry, rz]
        conversion_type: 转换类型，如 'left_to_right' 等
        angle_unit: 角度单位，'degrees' 或 'radians'
        apply_extra_rotation: 是否应用额外的绕 Y -90° 再绕 X -90° 旋转

    返回:
        [x_new, y_new, z_new, rx_new, ry_new, rz_new]: 转换后的坐标与旋转
    """
    x, y, z, rx, ry, rz = coord

    # 角度转弧度
    if angle_unit == 'degrees':
        rx, ry, rz = np.radians([rx, ry, rz])

    # 基础坐标系转换
    if conversion_type == 'left_to_right':
        x_new, y_new, z_new = x, y, -z
        rx_new, ry_new, rz_new = rx, ry, -rz
    else :
        raise ValueError(f"不支持的转换类型：{conversion_type}")

    # 应用额外旋转：先绕 Y 轴 -90°，再绕 X 轴 -90°
    if apply_extra_rotation:
        x_new, y_new, z_new, rx_new, ry_new, rz_new = apply_extra_rotation_transform(
            x_new, y_new, z_new, rx_new, ry_new, rz_new
        )

    # 弧度转角度
    if angle_unit == 'degrees':
        rx_new, ry_new, rz_new = np.degrees([rx_new, ry_new, rz_new])

    return [x_new, y_new, z_new, rx_new, ry_new, rz_new]

def apply_extra_rotation_transform(x, y, z, rx, ry, rz):
    """
    应用额外旋转：先绕 Y 轴 -90°，再绕 X 轴 -90°
    """
    # 构建原始旋转矩阵（XYZ 顺序）
    
    ori_matrix = np.eye(4)
    ori_matrix[:3,3] = x,y,z
    ori_matrix[:3,:3] = R.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
    new_matrix =rot2 @ rot1 @ ori_matrix #@ rot1 #@ rot2
    x, y, z = new_matrix[:3,3]
    rx_new, ry_new, rz_new = R.from_matrix(new_matrix[:3,:3]).as_euler('xyz', degrees=True)

    return x, y, z, rx_new, ry_new, rz_new
