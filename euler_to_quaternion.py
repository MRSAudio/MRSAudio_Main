import math

def euler_to_quaternion(yaw_deg, pitch_deg, roll_deg):
    """
    将欧拉角（Z-X-Y顺序）转换为四元数。
    参数顺序：航向角(yaw, 绕Z轴), 俯仰角(pitch, 绕X轴), 滚转角(roll, 绕Y轴)，单位为度。
    返回四元数 (w, x, y, z)。
    """
    # 将角度转换为弧度
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    
    # 绕Z轴的旋转（yaw）
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)
    q_z = (cy, 0.0, 0.0, sy)  # (w, x, y, z)
    
    # 绕X轴的旋转（pitch）
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    q_x = (cp, sp, 0.0, 0.0)
    
    # 绕Y轴的旋转（roll）
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    q_y = (cr, 0.0, sr, 0.0)
    
    # 四元数乘法函数
    def multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return (w, x, y, z)
    
    # 按顺序相乘：q_y * q_x * q_z
    q_temp = multiply(q_x, q_z)
    q = multiply(q_y, q_temp)
    
    return q