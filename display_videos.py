import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio
import io

def vid(path,dst_folder,video,preds,y_pred,start,threshold):
    cap = cv2.VideoCapture(path+video+".mp4")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)    
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(dst_folder +video+".mp4", fourcc, fps, (int(frame_width), int(frame_height)))
    mins = sec = 0
    period = '00:00'
    n_sample = start
    count = 0
    count_sample = 50

    #plot figure
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot() 
    plt.xlabel('Frame')
    plt.ylabel('Prediction')
    plt.title("Predictions over time")
    plt.axhline(y = threshold, color = 'r', linestyle = '--',label='Threshold = 0.5') #'-o',color= "b"
    plt.legend(loc='upper left')
    plt.ylim(0, 1)
    y_pos_line = ax1.plot([], [], '-o',color = 'b')
    x_coords=[0]
    y_coords=[y_pred[n_sample][1]]

    try:
        while True:                      
            if count % 50 == 0 and count != 0: #count one sample every 50 frames
                n_sample = (count // 50) + start
                x_coords.append(count_sample)
                y_coords.append(y_pred[n_sample][1])               
                count_sample += 50
            ret, frame = cap.read()
            font = cv2.FONT_HERSHEY_SIMPLEX

            if not ret:               
                break

            cfn = cap.get(1)

            #set up timer
            if int(cfn)%int(fps)==1:
                if sec > 59:
                    sec = 0
                    mins = mins+1
                period = "{:02d}:{:02d}".format(mins,sec)
                sec = sec + 1
            cv2.putText(frame,period, (1600,1000),font,3,(0,0,0),10)

            #Display prediction in the screen
            #if np.argmax(y_pred[n_sample]) == 0:
            if preds[n_sample] == 0:
                text = "No-action"
                cv2.putText(frame,text, (0,100),font,3,(255,0,0),10)
            else:
                text = "Shot"
                cv2.putText(frame,text, (0,100),font,3,(0,0,255),10)
            
            #Update plot
            y_pos_line[0].set_xdata(x_coords)
            y_pos_line[0].set_ydata(y_coords) 
            if count_sample >= 250:
                plt.xlim(count_sample-250, 250+count_sample)
            else:
                plt.xlim(0, 500)  
            fig.canvas.draw()
            fig.canvas.flush_events()

            #Insert plot in the image
            with io.BytesIO() as buff:
                fig.savefig(buff, format='png')
                buff.seek(0)
                img = plt.imread(buff)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img * 255).astype(np.uint8)
            img = cv2.resize(img,[480,360])
            height, width, _ = frame.shape
            h, w, _ = img.shape           
            frame[0:h, width-w:width, :] = img[:, :, :3]

            cv2.imshow("prediction",frame)
            cv2.waitKey(1)            
            out.write(frame)
            count += 1
    finally:
        cap.release()
        out.release()

