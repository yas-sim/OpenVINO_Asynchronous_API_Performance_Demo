import os, sys
import copy
import time
import queue
import threading
from logging import getLogger, DEBUG, INFO, WARNING, ERROR, CRITICAL 

logger = getLogger(__name__)
logger.setLevel(INFO)

# Workaround for very slow OpenCV camera opening by VideoCapture() function issue on Windows
if os.name == 'nt':
    logger.info('OS is a Windows. Applied OpenCV workaround.')
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import numpy as np
import openvino as ov
import openvino.properties as props 
import openvino.properties.hint as hints 


class DemoOpenVINO():
    def __init__(self):
        self.image = np.zeros((480,640,3), dtype=np.uint8)  # Input image for inference
        self.image_lock = threading.Lock()                  # Lock object to update the input image data
        self.queue_renderd_result = queue.Queue()
        self.queue_inference_result = queue.Queue()
        self.input_stream = None
        self.abort_flag = False
        self.ready = False
        self.time_last_callback = time.perf_counter()
        self.time_last_render_result_update = time.perf_counter()
        self.fps = 0
        self.input_source = 0
        self.input_source = 'classroom.mp4'
        self.input_source = 'head-pose-face-detection-female-and-male.mp4'


    def load_and_init_model(self, model_file_name:str, target_device:str='CPU'):
        self.ov_model = ov.Core().read_model(model_file_name)

        # obtain model input information
        self.input_shape = self.ov_model.inputs[0].shape
        self.input_name = list(self.ov_model.inputs[0].names)[0]
        self.input_n, self.input_c, self.input_h, self.input_w = self.input_shape

        # obtain model output information
        self.output_shape = self.ov_model.outputs[0].shape
        self.output_name = list(self.ov_model.outputs[0].names)[0]

        # OpenVINO performance optimize parameters and hints
        config={'CACHE_DIR':'./cache'}
        config.update({hints.performance_mode: hints.PerformanceMode.THROUGHPUT})
        config.update({hints.num_requests:"32"})            # number of request queue
        if target_device in ['CPU']:
            config.update({props.inference_num_threads: "16"})  # number of thread used by OpenVINO runtime
        config.update({props.num_streams: "8"})             # number of simultaneous inference request execution

        self.model = ov.compile_model(self.ov_model, device_name=target_device, config=config)

        # Create async queue for easy handling
        self.async_infer_queue = ov.AsyncInferQueue(model=self.model, jobs=0) # jobs=0 means, automatic
        self.async_infer_queue.set_callback(self.callback)


    def thread_render_result(self):
        img_h, img_w = self.image.shape[:2]
        while self.abort_flag == False:
            if self.image is None:
                continue
            if self.queue_renderd_result.empty():
                time.sleep(0.01)                    # neglegible (short) wait
                continue
            result = self.queue_renderd_result.get()
            img = result[0]

            disp_str = f'{self.fps:7.2f} FPS'
            font = cv2.FONT_HERSHEY_PLAIN
            scale = 4
            thickness = 4
            (w, h), baseline = cv2.getTextSize(disp_str, font, scale, thickness)
            cv2.putText(img, disp_str, (0, h + baseline), font, scale, (0,0,0), thickness)
            cv2.putText(img, disp_str, (0, h + baseline), font, scale, (0,255,0), thickness-2)

            cv2.imshow('image', img)
            key = cv2.waitKey(25)       # a little shorter than 1/30 sec
            if key in (27, ord('q'), ord('Q'), ord(' ')):   # 27 is ESC key
                self.abort_flag = True


    def open_input_stream(self):
        self.input_stream = cv2.VideoCapture(self.input_source)
    
    def get_input_frame(self):
        if self.input_stream == None:
            self.open_input_stream()
        sts, img = self.input_stream.read()
        # reopen the input stream (for movie inputs)
        if sts == False:
            self.input_stream.release()
            self.open_input_stream()
            sts, img = self.input_stream.read()
            if sts == False:
                logger.error('Failed to reopen input stream')
                self.abort_flag = True
                img = np.zeros((640,480,3), dtype=np.uint8)     # dummy
        img = cv2.resize(img, (640, 480))
        return img

    # thread to capture the input image for inference
    def thread_capture_image(self):
        # Open input media stream
        self.open_input_stream()
        if self.input_stream is not None:
            self.input_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.input_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
            self.input_stream.set(cv2.CAP_PROP_FPS, 30)
            self.ready = True
        else:
            logger.error('Failed to open input stream. Abort')
            self.abort_flag = True
            return
        while self.abort_flag == False:
            img = self.get_input_frame()
            self.image_lock.acquire()
            self.image = img
            self.image_lock.release()
            time.sleep(1/30)


    # input image preprocess
    def preprocess(self, img):
        tensor = cv2.resize(img, (self.input_w, self.input_h))
        tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = tensor[np.newaxis, :, :, :]
        return tensor


    def thread_postprocess(self):
        while self.abort_flag == False:
            while self.queue_inference_result.empty():
                time.sleep((1/30)/2)                # No inference result. Sleep for a short time
            infer_result, img = self.queue_inference_result.get()
            result_image = img.copy()
            img_h, img_w = img.shape[:2]
            faces = infer_result[0, 0]              # (1, 1, 200, 7) -> (200, 7)
            for face in faces:
                obj_id, obj_cls, conf, x0, y0, x1, y1 = face
                if conf < 0.6:                      # ignore low confidence results
                    continue
                x0 = int(x0 * img_w)
                y0 = int(y0 * img_h)
                x1 = int(x1 * img_w)
                y1 = int(y1 * img_h)
                cv2.rectangle(result_image, (x0, y0), (x1, y1), (0,255,0), 2)
                # send the rendered result every 1/30 sec only
                if time.perf_counter() - self.time_last_render_result_update > 1/30:  # check if 1/30 sec elapsed
                    self.time_last_render_result_update = time.perf_counter()
                    self.queue_renderd_result.put((result_image,))


    # callback function to receive the asynchronous inference result
    def callback(self, request, userdata):      # userdata == input image for inferencing
        res = list(request.results.values())[0]
        if time.perf_counter() - self.time_last_callback < 1/30:        # submit the result to the rendering thread every 1/30 sec only
            return
        self.time_last_callback = time.perf_counter()
        self.queue_inference_result.put((res, userdata))


    def infer(self):
        # get an image captured by the image capture thread (therefore, synchronization is required)
        self.image_lock.acquire()
        img = self.image.copy()
        self.image_lock.release()

        tensor = self.preprocess(img)

        # run asynchronous inference
        # AsyncInferQueue object checks the availability of request queue.
        # No need to check it by the user code.
        # (But it is possible to check the request queue availability by .is_ready() member func)
        self.async_infer_queue.start_async(inputs=tensor, userdata=img)



    def run(self):
        self.th_render = None
        self.th_postprocess = None
        self.th_input = None
    
        self.load_and_init_model('face-detection-0200.xml', 'GPU.0') # 'CPU', 'GPU', 'GPU.0', 'GPU.1', 'NPU', ...

        self.th_render       = threading.Thread(target=self.thread_render_result)
        self.th_postprocess  = threading.Thread(target=self.thread_postprocess)
        self.th_input        = threading.Thread(target=self.thread_capture_image)

        self.th_render.start()
        self.th_postprocess.start()
        self.th_input.start()

        # run inference and measure performance
        num_loop = 10
        while self.abort_flag == False:
            stime = time.perf_counter()
            for _ in range(num_loop):
                self.infer()
            etime = time.perf_counter()
            self.fps = 1/((etime-stime)/num_loop)


    def __del__(self):
        self.abort_flag = True
        cv2.destroyAllWindows()
        if self.th_input is not None:
            self.th_input.join()
            self.th_input = None
        if self.th_render is not None:
            self.th_render.join()
            self.th_render = None
        if self.th_postprocess is not None:
            self.th_postprocess.join()
            self.th_postprocess = None
        if self.input_stream is not None:
            self.input_stream.release()
            self.input_stream = None


if __name__ == '__main__':
    demo = DemoOpenVINO()
    demo.run()
