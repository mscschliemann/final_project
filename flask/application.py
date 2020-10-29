from flask import Flask, render_template, request, Response
from camera import VideoCamera
import threading
import concurrent.futures

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

def gen_boxed(camera):
    while True:
        frame = camera.get_boxed_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    # model = request.args['m']
    # with concurrent.futures.ThreadPoolExecutor() as executor1:
    #     future = executor1.submit(gen_boxed, VideoCamera())
    #     return_value = future.result()
        #print(return_value)
    # x = threading.Thread(target=thread_function, args=(1,))
    # x.start()
    return Response(gen_boxed(VideoCamera()),
                mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response(return_value,
    #             mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_thresh(camera):
    while True:
        frame = camera.get_thresh_frame(((100, 100), (400, 400)))
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed_thresh')
def video_feed_thresh():
    # with concurrent.futures.ThreadPoolExecutor() as executor2:
    #     future = executor2.submit(gen_thresh, VideoCamera())
    #     return_value = future.result()
    return Response(gen_thresh(VideoCamera()),
                mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    #app.run(port=80, debug=True, threaded=True)    
    app.run(port=80, threaded=True)