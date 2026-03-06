from flask import Flask, render_template, Response, request, jsonify
from camera_engine import CameraEngine
import time

app = Flask(__name__)
engine = CameraEngine()

@app.route('/')
def index():
    # 渲染带有 Tailwind界面的前端页面
    return render_template('index.html')

def generate_frames():
    """ 视频推流生成器 (MJPEG format) """
    while True:
        frame = engine.get_frame()
        if frame is None:
            # 休息一会直到帧开始输出
            time.sleep(0.1)
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_mask_frames():
    """ 纯黑色掩膜流推流器 """
    while True:
        frame = engine.get_mask_frame()
        if frame is None:
            time.sleep(0.1)
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mask_feed')
def mask_feed():
    return Response(generate_mask_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    success = engine.start()
    return jsonify({"status": "success" if success else "error", 
                    "message": "Camera started" if success else "Failed to start camera"})

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    engine.stop()
    return jsonify({"status": "success", "message": "Camera stopped"})

@app.route('/api/hsv/update', methods=['POST'])
def update_hsv():
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400
        
    engine.update_hsv(data)
    return jsonify({"status": "success", "message": "HSV updated successfully", "current": engine.hsv_params})

@app.route('/api/hsv/current', methods=['GET'])
def get_hsv():
    return jsonify(engine.hsv_params)

@app.route('/api/color/pick', methods=['POST'])
def pick_color():
    data = request.json
    if not data or 'x' not in data or 'y' not in data:
        return jsonify({"status": "error", "message": "Invalid coordinates"}), 400
    
    success, result = engine.pick_color(data['x'], data['y'])
    if success:
        return jsonify({"status": "success", "message": "Color picked and HSV updated", "current": result})
    else:
        return jsonify({"status": "error", "message": result}), 400

@app.route('/api/roi/update', methods=['POST'])
def update_roi():
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400
        
    if data.get('clear', False):
        engine.update_roi(None, None, None, None)
    else:
        req_keys = ['x1', 'y1', 'x2', 'y2']
        if not all(k in data for k in req_keys):
            return jsonify({"status": "error", "message": "Missing coordinates"}), 400
            
        engine.update_roi(data['x1'], data['y1'], data['x2'], data['y2'])
        
    return jsonify({"status": "success", "message": "ROI updated successfully"})

if __name__ == '__main__':
    # 我们运行在 5050 端口（5000 端口在 Mac 上经常被内置 AirPlay 占用）
    app.run(host='0.0.0.0', port=5050, threaded=True, debug=True)
