import cv2
import numpy as np
import os
import math
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

def mouse_callback(event, x, y, flags, param):
    global clicked_points, input_requested
    # 只有还没到输入状态时才允许点选
    if not input_requested:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked_points) < 2:
                clicked_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            clicked_points = [] # 清空点位重新选择

def main():
    global clicked_points, input_requested
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头 (Index 0)")
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
            if len(clicked_points) == 0:
                display = draw_chinese_text(display, "请左键点击目标物体的第一个端点 (A点)", (20, 40), (0, 255, 255), 35)
                
            elif len(clicked_points) == 1:
                display = draw_chinese_text(display, "请左键点击目标物体的第二个端点 (B点)", (20, 40), (0, 255, 255), 35)
                # 画 A 点
                cv2.circle(display, clicked_points[0], 6, (0, 0, 255), -1)
                display = draw_chinese_text(display, "A", (clicked_points[0][0]+10, clicked_points[0][1]-20), (0, 0, 255), 30)
                
            elif len(clicked_points) == 2:
                display = draw_chinese_text(display, '已锁定线段！请回到 "终端窗口" 输入真实毫米长度...', (20, 40), (0, 255, 0), 30)
                # 绘制两端点及连线
                cv2.circle(display, clicked_points[0], 6, (0, 0, 255), -1)
                cv2.circle(display, clicked_points[1], 6, (0, 0, 255), -1)
                cv2.line(display, clicked_points[0], clicked_points[1], (255, 0, 0), 2)
                
                display = draw_chinese_text(display, "A", (clicked_points[0][0]+10, clicked_points[0][1]-20), (0, 0, 255), 30)
                display = draw_chinese_text(display, "B", (clicked_points[1][0]+10, clicked_points[1][1]-20), (0, 0, 255), 30)
            
            # 屏幕底部辅助按键提示
            display = draw_chinese_text(display, "👉 操作: 鼠标左键点击 | 鼠标右键重置(清空) | 键盘 'q' 键退出", (10, h - 40), (0, 255, 0), 25)

            cv2.imshow('Calibration Tool', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
            # 当两个点都收集齐且还未发起输入询问时，启动流程
            if len(clicked_points) == 2 and not input_requested:
                input_requested = True
                # 先刷新一帧，把绿色恭喜文本绘出
                cv2.waitKey(200) 
                
                p1 = clicked_points[0]
                p2 = clicked_points[1]
                pixel_length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                
                print(f"\n[操作提示] 您点选的 A点 和 B点 之间的直线像素距离为: {pixel_length:.2f} px")
                val = input("请输入此线段在现实中的真实直线长度 (单位：毫米 mm，输入回车则取消重选): ")
                
                # 开始解析数值
                try:
                    if not val.strip():
                        print("取消本次测量，请在画面里点击鼠标右键重置点位。")
                        input_requested = False
                        clicked_points = []
                        continue
                        
                    phys_length = float(val)
                    if phys_length <= 0:
                        print("错误：距离必须大于0。")
                        input_requested = False
                        clicked_points = []
                        continue
                        
                    # 计算系数
                    pixels_per_mm = pixel_length / phys_length
                    
                    # 为了兼容后续去畸变程序的逻辑，我们提供一对标准的平庸参数
                    # (由于采用极简两点测量，此方法假定镜头本身的形变对长度影响可忽略)
                    mtx = np.eye(3, dtype=np.float32)
                    dist = np.zeros(5, dtype=np.float32)
                    
                    np.savez(PARAMS_FILE, camera_matrix=mtx, dist_coefs=dist, pixels_per_mm=pixels_per_mm)
                    
                    print("\n" + "="*50)
                    print(f"✅ 标定校准已成功！")
                    print(f"当前相机的物理转换比例为: {pixels_per_mm:.4f} pixels/mm")
                    print(f"换算结果文件已保存至项目内的 {PARAMS_FILE}")
                    print("现在您可以直接退出程序，并运行 measure_hose_line.py 啦！")
                    print("="*50 + "\n")
                    break # 成功即完成使命，退出循环
                    
                except ValueError:
                    print("输入无效！只允许输入纯数字格式。请重新选点测量。")
                    input_requested = False
                    clicked_points = []
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
