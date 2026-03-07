import cv2
import numpy as np
import os
import argparse
import math
from PIL import Image, ImageDraw, ImageFont

PARAMS_FILE = 'camera_params.npz'

# ----------------- 补偿与状态管理 -----------------
# 针对非工业远心镜头（普通广角摄像头）近距离微距引发的球面膨胀系数进行补丁
# 您可以不断在这里乘入新的校准比例。
# 第一次: 真实 17.0, 但画面原始拍出膨胀到 18.2 -> 比率 17.0 / 18.2 ≈ 0.934
# 第二次: 加上了 0.934 后, 测出一根线是 16.9, 但其实它是 16.2 -> 还需要再乘上 16.2 / 16.9
CAMERA_OPTICAL_SHRINK_RATIO = (17.0 / 18.2) * (16.2 / 16.9)

app_mode = 'STANDBY' # 阶段状态: 'STANDBY', 'PICK_COLOR', 'MEASURE'
drawing = False      # 是否正在拖拽鼠标圈定 ROI
ix, iy = -1, -1      # 鼠标按下的起点
curr_x, curr_y = 0,0 # 鼠标当前掠过的落点
roi_box = None       # 保存的 (x1, y1, x2, y2)
clean_frame = None  # 用于存放纯净的去畸变原图供“点击取色”分析

# 缓存字体实例，防止每帧都在硬盘寻址字体造成几百毫秒的卡顿掉帧
_cached_fonts = {}

def get_chinese_font(size):
    if size in _cached_fonts:
        return _cached_fonts[size]
        
    font_paths = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf"
    ]
    for path in font_paths:
        if os.path.exists(path):
            font = ImageFont.truetype(path, size)
            _cached_fonts[size] = font
            return font
            
    font = ImageFont.load_default()
    _cached_fonts[size] = font
    return font

def nothing(x):
    """ 滑动条的回调占位符 """
    pass

def load_calibration():
    """ 功能1：加载相机参数 """
    if not os.path.exists(PARAMS_FILE):
        print(f"警告: 找不到校准文件 '{PARAMS_FILE}'。系统不能根据物理尺寸换算。")
        return None, None, None
        
    try:
        data = np.load(PARAMS_FILE)
        mtx = data['camera_matrix']
        dist = data['dist_coefs']
        pixels_per_mm = float(data['pixels_per_mm'])
        return mtx, dist, pixels_per_mm
    except Exception as e:
        print(f"读取参数文件失败: {e}")
        return None, None, None

def set_trackbars(target_h, target_s, target_v):
    """ 根据提取到的指定颜色，配置推荐的容差滑动块 """
    
    # --- 针对【极端黑色/极深色】的特判 ---
    if target_v < 60 or (target_s < 40 and target_v < 90):
        # 黑色/深灰几乎没有色相(H)，只看亮度(V)必须低于阈值
        cv2.setTrackbarPos('H_Min', 'Control Panel', 0)
        cv2.setTrackbarPos('H_Max', 'Control Panel', 179)
        cv2.setTrackbarPos('S_Min', 'Control Panel', 0)
        cv2.setTrackbarPos('S_Max', 'Control Panel', 255)
        # 允许V从0(绝对黑)开始，上限放宽以包括反光灰
        cv2.setTrackbarPos('V_Min', 'Control Panel', 0)
        cv2.setTrackbarPos('V_Max', 'Control Panel', min(255, target_v + 80))
        return

    # --- 针对【正常彩色】的放宽容差 ---
    h_margin = 25
    s_margin = 150
    v_margin = 150
    
    cv2.setTrackbarPos('H_Min', 'Control Panel', max(0, target_h - h_margin))
    cv2.setTrackbarPos('H_Max', 'Control Panel', min(179, target_h + h_margin))
    cv2.setTrackbarPos('S_Min', 'Control Panel', max(10, target_s - s_margin))
    cv2.setTrackbarPos('S_Max', 'Control Panel', 255)
    cv2.setTrackbarPos('V_Min', 'Control Panel', max(20, target_v - v_margin))
    cv2.setTrackbarPos('V_Max', 'Control Panel', 255)

def mouse_callback(event, x, y, flags, param):
    """ 功能4：原生鼠标操作回调 """
    global drawing, ix, iy, curr_x, curr_y, roi_box, clean_frame, app_mode
    
    # 记录鼠标当前点，供画出高亮游标线框
    curr_x, curr_y = x, y
    
    if app_mode == 'PICK_COLOR':
        if event == cv2.EVENT_LBUTTONDOWN:
            # 单纯点击 -> 取色并填充滑块
            if clean_frame is not None:
                h, w = clean_frame.shape[:2]
                if 0 <= x < w and 0 <= y < h:
                    y1, y2 = max(0, y - 1), min(h, y + 2)
                    x1, x2 = max(0, x - 1), min(w, x + 2)
                    roi_region = clean_frame[y1:y2, x1:x2]
                    
                    mean_bgr = cv2.mean(roi_region)[:3]
                    bgr_pixel = np.uint8([[mean_bgr]])
                    hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]
                    
                    set_trackbars(int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2]))
                    print(f"[*] 取色成功：HSV={hsv_pixel}。请按键盘 'Enter' 或 'm' 键启动测长！")
                    roi_box = None

    elif app_mode == 'MEASURE':
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            dist = np.hypot(x - ix, y - iy)
            # 拖拽必须具备明确拉出面积的意图
            if dist > 5:
                roi_box = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))
                print(f"[*] 已在测长过程施加遮罩区 (ROI)：{roi_box}")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键点击 -> 清除选区 (ROI)
            roi_box = None
            print("[*] 已取消遮罩区限定。系统恢复全画面检测。")


# 之前的 get_chinese_font 已经被移到文件顶部加了缓存机制

def draw_chinese_text(img_bgr, text, position, text_color=(0, 255, 0), font_size=40):
    """ 功能3：中文显示支持 (使用 PIL 解决文字乱码) """
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = get_chinese_font(font_size)
    
    # 颜色顺序转换 (BGR -> RGB)
    rgb_color = (text_color[2], text_color[1], text_color[0])
    draw.text(position, text, fill=rgb_color, font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def main():
    global clean_frame, roi_box, drawing, ix, iy, app_mode
    
    parser = argparse.ArgumentParser(description="GD-VisionMark 高质量视觉测长程序")
    parser.add_argument('-c', '--camera', type=int, default=0, help="指定摄像头设备的编号索引 (默认: 0)")
    args = parser.parse_args()
    
    mtx, dist, pixels_per_mm = load_calibration()
    
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"错误: 无法打开指定的视频采集设备 (Index {args.camera})。")
        print("💡 提示：如果刚插入新外接摄像头，可以尝试在启动命令后加参数指定，例如: python3 measure_hose_line.py -c 1")
        return

    # 1. 独立开启一个带追踪条的控制器窗口（置于最顶端方便调节）
    cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Control Panel', 400, 300)
    cv2.moveWindow('Control Panel', 50, 50)
    
    # 填充默认的缺省范围
    cv2.createTrackbar('H_Min', 'Control Panel', 15, 179, nothing)
    cv2.createTrackbar('H_Max', 'Control Panel', 35, 179, nothing)
    cv2.createTrackbar('S_Min', 'Control Panel', 80, 255, nothing)
    cv2.createTrackbar('S_Max', 'Control Panel', 255, 255, nothing)
    cv2.createTrackbar('V_Min', 'Control Panel', 40, 255, nothing) # 防治过暗的噪点反光
    cv2.createTrackbar('V_Max', 'Control Panel', 255, 255, nothing)

    # 2. 独立开启主画面并挂载鼠标交互，占据主力视窗
    cv2.namedWindow('Main View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Main View', 1000, 750)
    cv2.moveWindow('Main View', 480, 50)
    cv2.setMouseCallback('Main View', mouse_callback)
    
    # 构建更锋利的形态学算法算子，断绝细碎反倒影响
    kernel_open = np.ones((7, 7), np.uint8)  # 大开断连
    # 大幅度强化闭运算，主要用于强行把同一条线因为严重高光或断色而分成两截的色块缝合在一起
    kernel_close = np.ones((9, 9), np.uint8)

    print("\n" + "="*50)
    print(" 已启动本地高性能测量程序！")
    print(" [工作流程]")
    print(" 1. 待机时按键盘 'c' 进入 [选色模式]。")
    print(" 2. 在画面中用左键单击管上的颜色带进行颜色提取。")
    print(" 3. 提取后按键盘 'Enter' 或 'm' 键锁定颜色并进入 [正式测长模式]。")
    print(" 4. 按键盘 'q' 键退出进程。")
    print("="*50 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 去除畸变逻辑
            if mtx is not None and dist is not None:
                undistorted = cv2.undistort(frame, mtx, dist, None, mtx)
            else:
                undistorted = frame.copy()
                
            # 存一份干净的像素副本供左键取色器分析
            clean_frame = undistorted.copy()
            display_frame = undistorted.copy()
            h, w = undistorted.shape[:2]
            
            if app_mode == 'STANDBY':
                display_frame = draw_chinese_text(display_frame, "准备就绪！请按键盘的 'c' 键进入 [选色模式]", (30, 60), (0, 255, 255), 50)
                
            elif app_mode == 'PICK_COLOR':
                # 绘制取色游标和提示
                cv2.line(display_frame, (0, curr_y), (w, curr_y), (150, 150, 150), 1)
                cv2.line(display_frame, (curr_x, 0), (curr_x, h), (150, 150, 150), 1)
                cv2.circle(display_frame, (curr_x, curr_y), 5, (0, 0, 255), 1)
                display_frame = draw_chinese_text(display_frame, "【选色模式】请点击目标色带取色。完毕后按 [Enter] 或 [m] 进入测长", (20, 50), (0, 255, 255), 35)

            elif app_mode == 'MEASURE':
                # 读取当前用户在 UI 滑动条设定的下限和上限值
                h_min = cv2.getTrackbarPos('H_Min', 'Control Panel')
                h_max = cv2.getTrackbarPos('H_Max', 'Control Panel')
                s_min = cv2.getTrackbarPos('S_Min', 'Control Panel')
                s_max = cv2.getTrackbarPos('S_Max', 'Control Panel')
                v_min = cv2.getTrackbarPos('V_Min', 'Control Panel')
                v_max = cv2.getTrackbarPos('V_Max', 'Control Panel')
                
                lower = np.array([h_min, s_min, v_min])
                upper = np.array([h_max, s_max, v_max])
                hsv = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HSV)
                
                # -> 寻找基础色带掩膜
                mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
                
                # -> [新增 ROI 限定拦截] <-
                if roi_box is not None:    
                    x1, y1, x2, y2 = roi_box
                    roi_mask = np.zeros_like(mask)
                    roi_mask[y1:y2, x1:x2] = 255     
                    mask = cv2.bitwise_and(mask, roi_mask)
                    
                # 从加工好的掩膜里寻找最大的白块连通轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 0:
                    max_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(max_contour) > 200:
                        # ---------------- 新一代外接矩形长宽解构测距法 ----------------
                        rect = cv2.minAreaRect(max_contour)
                        box = cv2.boxPoints(rect)
                        
                        edge01 = math.hypot(box[0][0]-box[1][0], box[0][1]-box[1][1])
                        edge12 = math.hypot(box[1][0]-box[2][0], box[1][1]-box[2][1])
                        
                        if edge01 > edge12:
                            # 边 01 长，说明短边是 12 和 03。连接这两条短边的中点
                            estimated_length_px = edge01
                            estimated_width_px = edge12
                            pt_start = ((box[1][0]+box[2][0])/2, (box[1][1]+box[2][1])/2)
                            pt_end   = ((box[0][0]+box[3][0])/2, (box[0][1]+box[3][1])/2)
                        else:
                            # 边 12 长，说明短边是 01 和 23。连接这两条短边的中点
                            estimated_length_px = edge12
                            estimated_width_px = edge01
                            pt_start = ((box[0][0]+box[1][0])/2, (box[0][1]+box[1][1])/2)
                            pt_end   = ((box[2][0]+box[3][0])/2, (box[2][1]+box[3][1])/2)
                            
                        pt_start, pt_end = (int(pt_start[0]), int(pt_start[1])), (int(pt_end[0]), int(pt_end[1]))
                        
                        if pixels_per_mm is not None and pixels_per_mm > 0:
                            # 原始像素距离除以标定比例
                            raw_len_mm = estimated_length_px / pixels_per_mm
                            raw_wid_mm = estimated_width_px / pixels_per_mm
                            
                            # 引入抗广角微距光学膨胀的补偿系数折算真实长度
                            phys_len_mm = raw_len_mm * CAMERA_OPTICAL_SHRINK_RATIO
                            phys_wid_mm = raw_wid_mm * CAMERA_OPTICAL_SHRINK_RATIO
                            
                            output_text = f"线长: {phys_len_mm:.1f} mm | 线宽: {phys_wid_mm:.1f} mm"
                            color = (0, 255, 0)
                        else:
                            output_text = f"未校准线长: {estimated_length_px:.1f} px"
                            color = (0, 255, 255)
                            
                        # ---------- 绘制层 ----------
                        cv2.drawContours(display_frame, [max_contour], -1, color, 2)
                        
                        box_int = np.int0(box)
                        cv2.drawContours(display_frame, [box_int], 0, (0, 0, 255), 2)
                        
                        cv2.line(display_frame, pt_start, pt_end, (255, 0, 255), 4)
                        
                        display_frame = draw_chinese_text(display_frame, output_text, (30, 60), color, 50)
                else:
                     display_frame = draw_chinese_text(display_frame, "等待目标...", (30, 60), (0, 165, 255), 50)
                
            # -> UI交互层的画面绘制：蓝色半透明拖曳区域或当前框定范围
            if drawing:
                # 正在绘画时跟随游标
                cv2.rectangle(display_frame, (ix, iy), (curr_x, curr_y), (255, 100, 100), 2)
            elif roi_box is not None:
                # 绘制最终锁定的 ROI
                x1, y1, x2, y2 = roi_box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # 左上角打个 Tag
                cv2.putText(display_frame, "ROI Mask Active", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
                
            # 屏幕下侧绘制按键提示 (BGR: Red, 更粗体)
            h, w = display_frame.shape[:2]
            cv2.putText(display_frame, "L-Click: Pick Color | Drag: Draw ROI | R-Click: Clear ROI | Q: Quit", (10, h-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # 仅推流主画面至屏幕，省去不必要的多开计算
            cv2.imshow('Main View', display_frame)
            
            # macOS 下防僵死关键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') or key == ord('C'):
                app_mode = 'PICK_COLOR'
                roi_box = None
            elif key == ord('m') or key == ord('M') or key == 13 or key == 10:
                if app_mode == 'PICK_COLOR':
                    app_mode = 'MEASURE'
                
    finally:
        # 清除句柄资源
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
