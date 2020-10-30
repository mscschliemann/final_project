from flask import Flask, render_template, request, Response
from camera import VideoCamera

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen_boxed(camera):
    while True:
        frame = camera.get_boxed_frame()
        ret = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        yield ret

@app.route('/video_feed')
def video_feed():
    frame = gen_boxed(VideoCamera())
    return Response(frame,
                mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_thresh(camera):
    while True:
        frame = camera.get_thresh_frame(((100, 100), (400, 400)))
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed_thresh')
def video_feed_thresh():
    return Response(gen_thresh(VideoCamera()),
                mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_game(camera, val):
    capture_rect = ((100, 100), (400, 400))
    player_result = ""
    computer_result = ""
    win_result = ""
    get_value = True
    while True:
        frame = camera.get_game_frame(capture_rect)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n',
               "Hallo")

@app.route('/video_game')
def video_feed_game():
    ret = "Hi"
    frame, ret = gen_game(VideoCamera(), ret)
    return Response(frame,
                mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    #app.run(port=80, debug=True, threaded=True)    
    app.run(port=80, threaded=True)