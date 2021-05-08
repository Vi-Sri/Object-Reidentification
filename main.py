import cv2
from multiprocessing import Process, Queue
from tflite_runtime.interpreter import Interpreter
import numpy as np
import os

BASE_PATH = "videos"
classes_ids = [42,40]
labels = {
    46 : "bottle",
    43 : "cup"
}

def frame_processor(video_file, frame_queue):
    interpreter = Interpreter(model_path="model/ssd_mobilenet_v1_1_metadata_1.tflite")
    interpreter.allocate_tensors()
    video = cv2.VideoCapture(os.path.join(BASE_PATH,video_file))
    imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    while(video.isOpened()):
        ret, frame = video.read()
        if not ret:
            print('Reached the end of the video!')
            frame_queue.put(None)
            break
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame_resized = cv2.resize(frame_rgb, (width, height))
        # input_data = np.expand_dims(frame_resized, axis=0)

        # interpreter.set_tensor(input_details[0]['index'],input_data)
        # interpreter.invoke()

        # boxes = interpreter.get_tensor(output_details[0]['index'])[0] 
        # classes = interpreter.get_tensor(output_details[1]['index'])[0]
        # scores = interpreter.get_tensor(output_details[2]['index'])[0]
        # #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        # for i in range(len(scores)):
        #     if (scores[i] >= 0.2):
        #         if int(classes[i]) == 46 or int(classes[i]) == 43:
        #             ymin = int(max(1,(boxes[i][0] * imH)))
        #             xmin = int(max(1,(boxes[i][1] * imW)))
        #             ymax = int(min(imH,(boxes[i][2] * imH)))
        #             xmax = int(min(imW,(boxes[i][3] * imW)))
        #             cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)
        #             # object_name = labels[int(classes[i])]
        #             object_name = "OOI"
        #             label = '%s: %d%%' % (object_name, int(scores[i]*100)) 
        #             labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        #             label_ymin = max(ymin, labelSize[1] + 10) 
        #             cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        #             cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        frame_queue.put(frame)

def kill_process(process_list):
    for p_elem in process_list:
        if p_elem[0].is_alive():
            p_elem[0].terminate()

def main():
    videos = ["1.mp4", "2.mp4", "3.mp4"]
    processList = []

    result = cv2.VideoWriter('res_01.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 30, (640, 480*3))

    for vfile in videos:
        frame_q = Queue(100)
        p_obj = Process(target=frame_processor, args=(vfile, frame_q))
        p_obj.start()
        processList.append([p_obj,frame_q])

    stop_flag_signal = []

    while True:
        frame_list = []
        for i in range(len(videos)):
            frame_i = processList[i][1].get()
            if frame_i is not None:
                frame_i = cv2.resize(frame_i, (640,480))
                frame_list.append(frame_i)
            else:
                stop_flag_signal.append(False)
                frame_list.append(np.zeros((480,640,3)))
        
        if all(stop_flag_signal) == False and len(stop_flag_signal) == len(videos):
            break
        
        print(frame_list[0].shape, frame_list[1].shape, frame_list[2].shape)
        stacked_frame = np.vstack(frame_list)
        print("going onn..")
        result.write(stacked_frame)
        # cv2.imshow("display_window", stacked_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # cv2.destroyAllWindows()
    kill_process(processList)


if __name__ == "__main__":
    main()
            

            
        
    


    
    
    

        
        
