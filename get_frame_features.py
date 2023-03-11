from tensorflow import keras
from glob import glob
import numpy as np
import cv2

MAX_SEQ_LENGTH = 55
NUM_FEATURES = 1024
WIDTH = 1920
HEIGHT = 1080

def load_video(path, video, start_frame,end_frame,max_frames=0): 
    cap = cv2.VideoCapture(path+video+".mp4") 
    frames = []
    count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)
    try:
        while True and count<end_frame-start_frame:
            ret, frame = cap.read()
            if not ret:
                print("break")
                break
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
            count = count + 1
    except:
        pass
    finally:
        cap.release()
    return np.array(frames)

def build_feature_extractor(): #Pretrained DenseNet
    feature_extractor = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(HEIGHT,WIDTH, 3),
    )
    preprocess_input = keras.applications.densenet.preprocess_input
    inputs = keras.Input((HEIGHT,WIDTH, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

def prepare_all_videos(path,videos,labels,start_frames,end_frames):
    frame_features = np.zeros(shape=(len(videos), MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    for idx,video in enumerate(videos):
        print(video)
        frames = load_video(path,video,start_frames[idx],end_frames[idx],max_frames=0) #convert video to array format
       
        # Pad shorter videos.
        if len(frames) < MAX_SEQ_LENGTH:
            if len(frames) == 0:
                frames = np.zeros((MAX_SEQ_LENGTH,HEIGHT,WIDTH, 3))
            else:
                diff = MAX_SEQ_LENGTH - len(frames)            
                padding = np.zeros((diff, HEIGHT,WIDTH, 3))           
                frames = np.concatenate((frames, padding))           

        frames = frames[None, ...]
        # Initialize placeholder to store the features of the current video.
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
        
        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                if np.mean(batch[j, :]) > 0.0:
                    temp_frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
                else:
                    temp_frame_features[i, j, :] = 0.0
        frame_features[idx,] = temp_frame_features.squeeze()
    return frame_features, labels

path= "videos/" #folder containing raw videos
input = np.load('data/clips_info.npy') #import list containing labels and time intervals of clips
videos = []
start_frames = []
end_frames = []
labels = []
for item in input:
    videos.append(item[0])
    labels.append(int(item[1]))
    start_frames.append(int(item[2]))
    end_frames.append(int(item[3]))

frame_features,labels = prepare_all_videos(path,videos,labels,start_frames,end_frames)
#np.save('frame_features.npy', frame_features)
#np.save('labels.npy', labels)
