import os, sys
import copy
import time
import queue
from logging import getLogger, DEBUG, INFO, WARNING, ERROR, CRITICAL 

logger = getLogger(__name__)
logger.setLevel(INFO)

# Workaround for very slow OpenCV VideoCapture() function issue on Windows platform 
if os.name == 'nt':
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import numpy as np
import openvino as ov
import openvino.properties as props 
import openvino.properties.hint as hints 


class DemoOpenVINO():
    def __init__(self):
        self.input_stream = None
        self.abort_flag = False
        self.ready = False
        self.fps = 0

        self.input_source = 0
        self.input_source = './classroom.mp4'
        self.input_source = 'head-pose-face-detection-female-and-male.mp4'
        self.open_input_stream()
        if self.input_stream is not None:
            self.input_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.input_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
            self.input_stream.set(cv2.CAP_PROP_FPS, 30)
            self.ready = True
            self.time_last_update = time.perf_counter()
        else:
            logger.error('Failed to open input stream. Abort')
            self.abort_flag = True
            return


    def load_and_init_model(self, model_file_name):
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
        self.model = ov.compile_model(self.ov_model, device_name='CPU', config=config)

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

    def preprocess(self, img):
        tensor = cv2.resize(img, (self.input_w, self.input_h))
        tensor = cv2.cvtColor(tensor, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = tensor[np.newaxis, :, :, :]
        return tensor


    def postprocess(self, img, infer_result):
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
            cv2.rectangle(img, (x0, y0), (x1, y1), (0,255,0), 2)

        disp_str = f'{self.fps:7.2f} FPS'
        font = cv2.FONT_HERSHEY_PLAIN
        scale = 4
        thickness = 4
        (w, h), baseline = cv2.getTextSize(disp_str, font, scale, thickness)
        cv2.putText(img, disp_str, (0, h + baseline), font, scale, (0,0,0), thickness)
        cv2.putText(img, disp_str, (0, h + baseline), font, scale, (0,255,0), thickness-2)

        cv2.imshow('image', img)
        key = cv2.waitKey(1)
        if key in (27, ord('q'), ord('Q'), ord(' ')):   # 27 is ESC key
            self.abort_flag = True

        return img


    def infer(self):
        img = self.get_input_frame()
        tensor = self.preprocess(img)

        res = self.model(tensor)

        infer_result = res[self.output_name]
        self.postprocess(img, infer_result)


    def run(self):
        self.load_and_init_model('face-detection-0200.xml')

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
        if self.input_stream is not None:
            self.input_stream.release()
            self.input_stream = None


if __name__ == '__main__':
    demo = DemoOpenVINO()
    demo.run()
