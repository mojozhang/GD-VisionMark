import cv2
import numpy as np
import os
import math
import argparse
from PIL import Image, ImageDraw, ImageFont

PARAMS_FILE = 'camera_params.npz'

def get_chinese_font(size):
    font_paths = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf"
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def draw_chinese_text(img_bgr, text, position, text_color=(0, 255, 0), font_size=30):
    """ 使用 PIL 解决 OpenCV 原生绘制中文乱码的问题 """
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = get_chinese_font(font_size)
    rgb_color = (text_color[2], text_color[1], text_color[0])
    draw.text(position, text, fill=rgb_color, font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# 全局变量保存点击的点
clicked_points = []
input_requested = False
curr_x, curr_y = 0, 0
input_string = ""

def mouse_callback(event, x, y, flags, param):
    global clicked_points, input_requested, curr_x, curr_y, input_string
    # 无论何时，更新当前鼠标坐标
    curr_x, curr_y = x, y
    
    # 只有还没到输入状态时才允许点选
    if not input_requested:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked_points) < 2:
                clicked_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            clicked_points = [] # 清空点位重新选择
            input_string = ""
    else:
        # 输入状态下点右键，说明想取消本次量测
        if event == cv2.EVENT_RBUTTONDOWN:
            clicked_points = []
            input_requested = False
            input_string = ""

def main():
    global clicked_points, input_requested, input_string
    
    parser = argparse.ArgumentParser(description="GD-VisionMark 相机比例物理标定程序")
    parser.add_argument('-c', '--camera', type=int, default=0, help="指定摄像头设备的编号索引 (默认: 0)")
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 (Index {args.camera})")
        print("💡 提示：如果刚插入新外接摄像头，可以尝试在启动命令后加参数指定，例如: python3 calibrate_camera.py -c 1")
        return
        
    cv2.namedWindow('Calibration Tool', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Calibration Tool', mouse_callback)

    print("\n" + "="*50)
    print(" 已启动 [极简两点标定法]！")
    print(" 1. 请在弹出的视频窗口中，用鼠标分别点击已知长度物体的两个端点。")
    print(" 2. 当点满两点后，终端里会要求您输入实际的物理距离 (单位: 毫米)。")
    print("="*50 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("摄像头读取失败。")
                break
                
            display = frame.copy()
            h, w = display.shape[:2]
            
            # --- 渲染中文画面提示 ---
            # 绘制全屏辅助十字准星（防止看不清或者手抖）
            if not input_requested:
                cv2.line(display, (0, curr_y), (w, curr_y), (150, 150, 150), 1)
                cv2.line(display, (curr_x, 0), (curr_x, h), (150, 150, 150), 1)

            if len(clicked_points) == 0:
                display = draw_chinese_text(display, "请使用十字准星对准并左键点击第一个端点 (A点)", (20, 40), (0, 255, 255), 35)
                
            elif len(clicked_points) == 1:
                display = draw_chinese_text(display, "请左键点击目标物体的第二个端点 (B点)", (20, 40), (0, 255, 255), 35)
                # 画 A 点准星 (极细的十字和空心小圆)
                cv2.circle(display, clicked_points[0], 3, (0, 0, 255), 1)
                cv2.line(display, (clicked_points[0][0]-10, clicked_points[0][1]), (clicked_points[0][0]+10, clicked_points[0][1]), (0, 255, 0), 1)
                cv2.line(display, (clicked_points[0][0], clicked_points[0][1]-10), (clicked_points[0][0], clicked_points[0][1]+10), (0, 255, 0), 1)
                
                display = draw_chinese_text(display, "A", (clicked_points[0][0]+10, clicked_points[0][1]-20), (0, 0, 255), 30)
                
            elif len(clicked_points) == 2:
                # 画两端点准星
                for pt in clicked_points:
                    cv2.circle(display, pt, 3, (0, 0, 255), 1)
                    cv2.line(display, (pt[0]-10, pt[1]), (pt[0]+10, pt[1]), (0, 255, 0), 1)
                    cv2.line(display, (pt[0], pt[1]-10), (pt[0], pt[1]+10), (0, 255, 0), 1)
                    
                # 绘制两端点精细连线
                cv2.line(display, clicked_points[0], clicked_points[1], (255, 0, 0), 1)
                
                display = draw_chinese_text(display, "A", (clicked_points[0][0]+10, clicked_points[0][1]-20), (0, 0, 255), 30)
                display = draw_chinese_text(display, "B", (clicked_points[1][0]+10, clicked_points[1][1]-20), (0, 0, 255), 30)
                
                p1 = clicked_points[0]
                p2 = clicked_points[1]
                pixel_length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                
                display = draw_chinese_text(display, f"已锁定！像素距 = {pixel_length:.1f} px", (20, 40), (0, 255, 0), 30)
                display = draw_chinese_text(display, f"请直接在键盘敲击对应纯数字物理长(mm)：{input_string}_", (20, 90), (0, 255, 255), 35)
                display = draw_chinese_text(display, "敲击 Enter(回车) 保存，或鼠标右键取消重选", (20, 140), (0, 200, 255), 25)
            
            # 屏幕底部辅助按键提示
            display = draw_chinese_text(display, "👉 操作: 鼠标左键点击 | 鼠标右键重置(清空) | 键盘 'q' 键退出", (10, h - 40), (0, 255, 0), 25)

            cv2.imshow('Calibration Tool', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
            # 当两个点都收集齐的时候，在画面内接管键盘输入字符（不再使用终端）
            if len(clicked_points) == 2:
                input_requested = True # 表示进入阻塞锁死输入态
                
                # 0-9 键 或者是 标点 '.' 键
                if (ord('0') <= key <= ord('9')) or key == ord('.'):
                    input_string += chr(key)
                # macOS 下的退格键可能是 8 或者 127
                elif key == 8 or key == 127:
                    input_string = input_string[:-1]
                # 回车键 13 或 10
                elif key == 13 or key == 10:
                    if input_string:
                        try:
                            phys_length = float(input_string)
                            if phys_length <= 0:
                                print("数值必须大于0！")
                                input_string = ""
                            else:
                                pixels_per_mm = pixel_length / phys_length
                                mtx = np.eye(3, dtype=np.float32)
                                dist = np.zeros(5, dtype=np.float32)
                                np.savez(PARAMS_FILE, camera_matrix=mtx, dist_coefs=dist, pixels_per_mm=pixels_per_mm)
                                print("\n" + "="*50)
                                print(f"✅ 标定校准已成功！")
                                print(f"当前相机的物理转换比例为: {pixels_per_mm:.4f} pixels/mm")
                                print(f"换算结果文件已保存至项目内的 {PARAMS_FILE}")
                                print("现在您可以直接退出程序，并运行 measure_hose_line.py 啦！")
                                print("="*50 + "\n")
                                break # 退出循环
                        except ValueError:
                            print("输入无效！只允许纯数字。")
                            input_string = ""
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
