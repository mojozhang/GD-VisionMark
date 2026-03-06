import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

PARAMS_FILE = 'camera_params.npz'

# ----------------- 全局状态管理 -----------------
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
    # 放宽色相容差，应对部分泛白的反光偏色
    h_margin = 20
    # 大幅放宽饱和度和亮度的下限容差，兼容褪色、变暗、灯光不均的真实情况
    s_margin = 120
    v_margin = 130
    
    cv2.setTrackbarPos('H_Min', 'Control Panel', max(0, target_h - h_margin))
    cv2.setTrackbarPos('H_Max', 'Control Panel', min(179, target_h + h_margin))
    # 工业现场可能有很多暗部，下限给得尽可能宽以增加容错
    cv2.setTrackbarPos('S_Min', 'Control Panel', max(10, target_s - s_margin))
    cv2.setTrackbarPos('S_Max', 'Control Panel', min(255, target_s + s_margin))
    cv2.setTrackbarPos('V_Min', 'Control Panel', max(20, target_v - v_margin))
    cv2.setTrackbarPos('V_Max', 'Control Panel', min(255, target_v + v_margin))

def mouse_callback(event, x, y, flags, param):
    """ 功能4：原生鼠标操作回调 """
    global drawing, ix, iy, curr_x, curr_y, roi_box, clean_frame
    
    # 记录鼠标当前点，供画出高亮游标线框
    curr_x, curr_y = x, y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 左键按下，准备拖动
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_LBUTTONUP:
        # 左键松开，判断是单纯点击，还是框选
        drawing = False
        dist = np.hypot(x - ix, y - iy)
        
        # 放大单纯“点击取色”的容忍脱手位移到 15 像素（防手抖变画框）
        if dist < 15:
            # 单纯点击 -> 取色并填充滑块
            if clean_frame is not None:
                h, w = clean_frame.shape[:2]
                if 0 <= x < w and 0 <= y < h:
                    # 获取该区块周围 3x3 进行求稳定均值
                    y1, y2 = max(0, y - 1), min(h, y + 2)
                    x1, x2 = max(0, x - 1), min(w, x + 2)
                    roi_region = clean_frame[y1:y2, x1:x2]
                    
                    mean_bgr = cv2.mean(roi_region)[:3]
                    bgr_pixel = np.uint8([[mean_bgr]])
                    hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]
                    
                    # 取色成功，强行调整控制面板
                    set_trackbars(int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2]))
                    print(f"[*] 取色成功：HSV=[{int(hsv_pixel[0])}, {int(hsv_pixel[1])}, {int(hsv_pixel[2])}]。已为您推算好容差。")
                    
                    # 重新清除 ROI 防止取色受限
                    roi_box = None
        else:
            # 拖拽必须具备明确拉出面积的意图 (边长不能过分小甚至反向拉成线) -> 固定 ROI 范围
            # 保证拉出来的框不是一条线
            if abs(x - ix) > 5 and abs(y - iy) > 5:
                roi_box = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))
                print(f"[*] 已施加遮罩区 (ROI) 过滤：{roi_box}")
            else:
                print(f"[*] 选取的框横竖太小，已被忽略，不建立 ROI 过滤区。")
            
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右键点击 -> 清除选区 (ROI)
        roi_box = None
        print("[*] 已取消遮罩区 (ROI) 限定。系统恢复全画面检测。")


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
    global clean_frame, roi_box, drawing, ix, iy
    
    mtx, dist, pixels_per_mm = load_calibration()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开指定的视频采集设备 (Index 0)。")
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
    print(" [操作指南]")
    print(" 1. 鼠标左键点击画面: 自动提取黄线色彩并设入面板。")
    print(" 2. 鼠标左键划出框选: 圈定 ROI 检测屏蔽外来干扰。")
    print(" 3. 鼠标右键点击画面: 取消一切框选 ROI。")
    print(" 4. 按键盘 'q' 键: 安全退出此程序。")
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
                # 只有原色带存在且同在框内才保留为白色
                mask = cv2.bitwise_and(mask, roi_mask)
                
            # 从加工好的掩膜里寻找最大的白块连通轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 作为显示的画布拷贝
            display_frame = undistorted.copy()
            
            # 绘制处理逻辑
            if len(contours) > 0:
                max_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(max_contour) > 200:
                    # 使用最小外接矩形法（能抗击边缘锯齿或坑洼带来的周长膨胀）
                    # minAreaRect 返回: ((center_x, center_y), (width, height), angle)
                    rect = cv2.minAreaRect(max_contour)
                    (w, h) = rect[1]
                    
                    # 取矩形的长边作为这根线的测量长度
                    estimated_length_px = max(w, h)
                    
                    if pixels_per_mm is not None and pixels_per_mm > 0:
                        phys_len_mm = estimated_length_px / pixels_per_mm
                        output_text = f"测量线长: {phys_len_mm:.1f} mm"
                        color = (0, 255, 0) # BGR 绿
                    else:
                        output_text = f"未校准线长: {estimated_length_px:.1f} px"
                        color = (0, 255, 255) # BGR 黄
                        
                    # 绘制检测信息
                    # 1. 依然绘制它原始不规则的连通轮廓（绿色）
                    cv2.drawContours(display_frame, [max_contour], -1, color, 3)
                    
                    # 2. 额外绘制用于实际物理测算依据的那个“最小外接倾斜矩形”（红色）
                    box = cv2.boxPoints(rect)
                    box = np.int0(box) # 转换为整数坐标
                    cv2.drawContours(display_frame, [box], 0, (0, 0, 255), 2)
                    
                    display_frame = draw_chinese_text(display_frame, output_text, (30, 60), color, 60)
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # 清除句柄资源
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
