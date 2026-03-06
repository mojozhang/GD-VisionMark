import os
import cv2
import numpy as np
import threading
from PIL import Image, ImageDraw, ImageFont

class CameraEngine:
    def __init__(self, params_file='camera_params.npz'):
        self.params_file = params_file
        self.mtx = None
        self.dist = None
        self.pixels_per_mm = None
        
        # 画面尺寸
        self.frame_width = 800
        self.frame_height = 600
        
        # 默认 HSV 参数
        self.hsv_params = {
            'h_min': 15, 'h_max': 35,
            's_min': 80, 's_max': 255,
            'v_min': 80, 'v_max': 255
        }
        
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # 用于保存最新处理过的一帧（JPEG 格式二进制数据）
        self.latest_frame = None
        # 用于保存黑白掩膜流数据 (JPEG 二进制)
        self.mask_frame = None
        
        # 用于保存纯净的没画线的原始帧（Numpy 数组形式），用于取色
        self.clean_frame = None
        
        # 用于记录前端传下的检测区域 (ROI)，格式为百分比 (x1, y1, x2, y2)
        self.roi_ratios = None
        
        self.load_calibration()
        
    def load_calibration(self):
        if not os.path.exists(self.params_file):
            print(f"警告: 找不到 '{self.params_file}'。画面将无法根据物理尺寸换算。请先运行标定脚本。")
            return
        try:
            data = np.load(self.params_file)
            self.mtx = data['camera_matrix']
            self.dist = data['dist_coefs']
            self.pixels_per_mm = float(data['pixels_per_mm'])
            print(f"加载校准参数成功，比例: {self.pixels_per_mm:.4f} px/mm")
        except Exception as e:
            print(f"读取参数文件出错: {e}")

    def start(self, camera_index=0):
        with self.lock:
            if self.is_running:
                return True
            
            self.cap = cv2.VideoCapture(camera_index)
            # Mac 摄像头建议指定默认读取尺寸以提升性能和统一输出排版
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            if not self.cap.isOpened():
                print("错误：无法开启摄像头")
                return False
                
            self.is_running = True
            
        # 启动后台常驻线程抓取处理视频
        thread = threading.Thread(target=self._capture_loop, daemon=True)
        thread.start()
        return True

    def stop(self):
        with self.lock:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None

    def update_roi(self, x1, y1, x2, y2):
        """ 接收四个顶点坐标比例以设置范围拦截。传入全 None 则表示清除界限 """
        with self.lock:
            if x1 is None:
                self.roi_ratios = None
            else:
                self.roi_ratios = (x1, y1, x2, y2)

    def update_hsv(self, params):
        with self.lock:
            for k, v in params.items():
                if k in self.hsv_params:
                    # 确保是 int
                    self.hsv_params[k] = int(v)

    def _capture_loop(self):
        kernel = np.ones((5, 5), np.uint8)
        
        while self.is_running:
            if self.cap is None:
                break
                
            ret, frame = self.cap.read()
            if not ret:
                continue

            # 1. 消除畸变
            if self.mtx is not None and self.dist is not None:
                display_frame = cv2.undistort(frame, self.mtx, self.dist, None, self.mtx)
            else:
                display_frame = frame.copy()
            
            # 存下一份纯净图像供外部取色器调用
            with self.lock:
                self.clean_frame = display_frame.copy()
            
            # 使用本地属性避免字典在别的线程被突变
            with self.lock:
                h_min = self.hsv_params['h_min']
                h_max = self.hsv_params['h_max']
                s_min = self.hsv_params['s_min']
                s_max = self.hsv_params['s_max']
                v_min = self.hsv_params['v_min']
                v_max = self.hsv_params['v_max']
                
            # 2. RGB 到 HSV
            hsv = cv2.cvtColor(display_frame, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([h_min, s_min, v_min])
            upper_yellow = np.array([h_max, s_max, v_max])
            
            # 3. 掩膜及形态学操作
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # 使用更大的腐蚀(开运算)和更小的膨胀(闭运算)，将外挂刺角和微弱倒影掐断剥离出来！
            kernel_open = np.ones((7, 7), np.uint8)  # 用大一点的核断开微弱连接处
            kernel_close = np.ones((3, 3), np.uint8) # 用小一点的核来弥补内部的断点即可
            
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
            
            # 推流其黑白掩膜本身，以便用户能够看到什么被留下了什么被抛弃了
            ret_mask, buffer_mask = cv2.imencode('.jpg', mask, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret_mask:
                self.mask_frame = buffer_mask.tobytes()
            
            # -> [新增 ROI 限定拦截] <-
            # 在寻找轮廓之前，如果有划定 ROI，则彻底擦除 ROI 外部的掩膜白块
            roi_active = False
            with self.lock:
                if self.roi_ratios is not None:
                    roi_active = True
                    rx1, ry1, rx2, ry2 = self.roi_ratios
            
            if roi_active:
                mh, mw = mask.shape
                # 比例转像素坐标
                px1, py1 = int(rx1 * mw), int(ry1 * mh)
                px2, py2 = int(rx2 * mw), int(ry2 * mh)
                
                # 排序保证格式正确
                x_start, x_end = min(px1, px2), max(px1, px2)
                y_start, y_end = min(py1, py2), max(py1, py2)
                
                # 创建一个全黑的底布
                roi_mask = np.zeros((mh, mw), dtype=np.uint8)
                # 将用户拉取到的矩形范围填白
                roi_mask[y_start:y_end, x_start:x_end] = 255
                
                # 两张图做 AND 运算，保留重叠的白色部分 (即选区内的原有黄线)，其余一切抹黑！
                mask = cv2.bitwise_and(mask, roi_mask)
                
                # 在画面上给个明显的蓝框提示用户目前的检测界限
                cv2.rectangle(display_frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            
            # 4. 寻找轮廓及计算长度
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                max_contour = max(contours, key=cv2.contourArea)
                
                # 更严格的面积控制，防止捡到飞边噪点
                if cv2.contourArea(max_contour) > 200:
                    arc_len_px = cv2.arcLength(max_contour, closed=False)
                    estimated_length_px = arc_len_px / 2.0
                    
                    if self.pixels_per_mm is not None and self.pixels_per_mm > 0:
                        phys_len_mm = estimated_length_px / self.pixels_per_mm
                        output_text = f"测量线长: {phys_len_mm:.1f} mm"
                        color = (0, 255, 0) # BGR 绿色
                    else:
                        output_text = f"未校准线长: {estimated_length_px:.1f} px"
                        color = (0, 255, 255) # BGR 黄色
                        
                    # OpenCV 画图（非文字）
                    cv2.drawContours(display_frame, [max_contour], -1, color, 2)
                    
                    # 使用 PIL 绘制中文以解决问号乱码
                    pil_img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                    try:
                        # Mac 常用自带的中文字体：PingFang.ttc ；Arial 不包含中文字形
                        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 40)
                    except IOError:
                        font = ImageFont.load_default()
                        
                    # 注意，PIL的颜色顺序是RGB，不是BGR，所以调整颜色(R=255, G=0, B=0) 为红
                    draw.text((30, 60), output_text, fill=(255, 0, 0), font=font)
                    display_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            else:
                # 使用 PIL 绘制中文
                pil_img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 36)
                except IOError:
                    font = ImageFont.load_default()
                # 橙色：R=255, G=165, B=0
                draw.text((30, 60), "等待目标...", fill=(255, 165, 0), font=font)
                display_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # 5. JPEG 编码用于 Web 推流
            ret, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                self.latest_frame = buffer.tobytes()

    def get_frame(self):
        return self.latest_frame
        
    def get_mask_frame(self):
        return self.mask_frame

    def pick_color(self, x_percent, y_percent):
        """
        根据前端传来的相对坐标比例 (0.0~1.0)，在原图中读取 BGR，并分析出一个合适的 HSV 范围
        返回 (bool, dict/str)
        """
        with self.lock:
            if self.clean_frame is None:
                return False, "还没有捕获到画面，请先启动监控"
            
            h, w = self.clean_frame.shape[:2]
            
            # 将相对百分比转化回绝对像素坐标
            px = int(x_percent * w)
            py = int(y_percent * h)
            
            # 越界保护
            if px < 0 or px >= w or py < 0 or py >= h:
                return False, "点击位置越狱"
                
            # 提取 3x3 小区块的中值颜色，防止正好点在单个噪点上
            y1, y2 = max(0, py - 1), min(h, py + 2)
            x1, x2 = max(0, px - 1), min(w, px + 2)
            roi = self.clean_frame[y1:y2, x1:x2]
            
            # 算平均 BGR
            mean_bgr = cv2.mean(roi)[:3] # 只取B,G,R三个通道的平均
            
            # 将平均 BGR 封存在 1x1 Numpy 里丢给 opencv 转 HSV
            bgr_pixel = np.uint8([[mean_bgr]])
            hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)
            
            target_h = int(hsv_pixel[0][0][0])
            target_s = int(hsv_pixel[0][0][1])
            target_v = int(hsv_pixel[0][0][2])
            
            # 我们给拾取的颜色划定一个“容差范围”窗口，保证识别一整条相近黄线
            # H 容差小一些 (颜色种类不能偏)，S和V 的容差大一些 (深浅亮度随意)。
            h_margin = 15
            s_margin = 60
            v_margin = 80
            
            # OpenCv 里的 H 最大 179，其余 255
            new_params = {
                'h_min': max(0, target_h - h_margin),
                'h_max': min(179, target_h + h_margin),
                's_min': max(20, target_s - s_margin), # 最低得有点颜色
                's_max': min(255, target_s + s_margin),
                'v_min': max(20, target_v - v_margin), # 不要纯黑色
                'v_max': min(255, target_v + v_margin)
            }
            
            # 自动应用这个新参数
            for k in new_params:
                self.hsv_params[k] = new_params[k]
                
            return True, self.hsv_params
