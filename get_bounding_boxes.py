from importlib.resources import files
from traceback import format_exception_only
from numpy import loadtxt
import os
import numpy as np
import cv2
import random
from glob import glob
import pandas as pd
import json

def get_bb(name,label,frame,labels_folder,path): #returns bounding box information 
    """
    Input: video name, action label, frame, labels folder, and its path
    Output: bounding box data, clip_info: [Name,label,start frame, end frame]
    """
    #get time interval of the clip
    if frame>5:
        frames = list(range(frame-5, frame+51))      
    else:
        frames = list(range(frame, frame+56))  

    #clip_info = [name,label,frames[0],frames[-1:][0]]
    data = [name,label,frames[0],frames[-1:][0]]
    frame = 0
    for i in frames: 
        file = path + "\\"+name+"_"+str(i+1)+".txt" #access corresponding bounding box file   
        if file in labels_folder:            
            big_goal = big_ball = big_keeper = 0
            big_players = []
            players = []
            ball = []
            goal = []
            keeper = []
            with open(file) as f:                          
                for line in f: # read rest of lines                 
                    new_players = [3]
                    line = line.split(" ")                                           
                    if line[0]=="6" and float(line[3])*float(line[4])>big_ball:  #check if object is a ball and if it is the biggest one            
                        ball = [0]
                        for x in line[1:5]: 
                            ball.append(float(x))
                        big_ball = float(line[3])*float(line[4])
                    elif line[0]=="8" and float(line[3])*float(line[4])>big_goal:                  
                        goal = [1]
                        for x in line[1:5]: 
                            goal.append(float(x))
                        big_goal = float(line[3])*float(line[4])                       
                    elif line[0]=="11" and float(line[3])*float(line[4])>big_keeper:                    
                        keeper = [2]
                        for x in line[1:5]: 
                            keeper.append(float(x))
                        big_keeper = float(line[3])*float(line[4])                   
                    elif line[0]=="7":
                        if len(big_players)>2:
                            if float(line[3])*float(line[4])>min(big_players): #check if player bb is one of the 3 biggest
                                index = big_players.index(min(big_players))
                                del big_players[index]
                                del players[index]                             
                                for x in line[1:5]: 
                                    new_players.append(float(x))
                                big_players.append(float(line[3])*float(line[4]))
                                players = players + [new_players]
                        else:
                            for x in line[1:5]: 
                                new_players.append(float(x))
                            big_players.append(float(line[3])*float(line[4]))
                            players = players + [new_players]                        
                #in case object is not detected
                if len(ball) == 0:
                    ball = [0,0,0,0,0] 
                if len(goal) == 0:
                    goal = [1,0,0,0,0]
                if len(keeper) == 0:
                    keeper = [2,0,0,0,0]                    
                if len(players) == 5:
                       players = [players]
                if len(players)<3:                    
                    for i in range(3-len(players)):
                        players.append([3,0,0,0,0])
                players_0 = players[0]
                players_1 = players[1]
                players_2 = players[2]
                all_objects = [ball,goal,keeper,players_0,players_1,players_2]
                data.append(all_objects)                           
            frame = frame + 1                          
        else: #in case file does not exist (video does not have more frames)
            break
    return data,data[:4]

def get_action_labels(file): #returns action label
    """
    Input: file
    Output: [Name,label,frame] of the clip
    """
    clip_info = []            
    f = open(file)
    data = json.load(f) 
    if data["annotated"] == True: #check if video contains actions      
        name = data["metadata"]["system"]["originalname"]
        num_labels = data["annotationsCount"] #get number of actions
        if num_labels == 1:
            d = data["annotations"]
            for x in d:
                label = x["label"]            
                frame = x["metadata"]["system"]["frame"] 
            clip_info = [name,label,frame]         
        else:
            frame_before = []
            for i in range(num_labels):
                d = data["annotations"]
                label = d[i]["label"]
                frame = d[i]["metadata"]["system"]["frame"]
                set = True
                for j in frame_before: #check if there are 2 overlapping actions (only keep the first one)
                    if abs(frame-j) < 50:
                        set = False
                        break
                if set == True:
                    new = [name,label,frame]               
                    clip_info = clip_info + new
                frame_before.append(frame)
    return clip_info

def get_no_action(file, all_clips_info,labels_folder,path): #get samples with no action
    """
    Input: file, list containing information about each clip already in the dataset, labels folder, and its path
    Output: bounding box data of the no-action sample, clip information: [Name,label (no-action), start frame, end frame]
    """
    f = open(file)
    data = json.load(f) 
    try:
        name = data["metadata"]["system"]["originalname"]
        name = name.split(".webm")[0]
        frames = data["metadata"]["system"]["ffmpeg"]["nb_read_frames"] #get total number of frames
        all_clips_names = [i[0] for i in all_clips_info] #get all the names of the used clips
        start = [i[2] for i in all_clips_info] #frame where clip starts
        end = [i[3] for i in all_clips_info] #frame where clip ends
        frame = random.randint(0,int(frames)) #select a random frame where the no-action clip will start
        index = 0
        for clip_name in all_clips_names:
            set = True
            if name in clip_name and abs(frame-int(start[index]))<=65: #check if the chosen video contains an action inside the chosen time interval
                set = False
                break 
            index = index + 1                                            
        label = "no-action"
        if set == True: #if clip does not contain an action
            sample_bb,clip_info = get_bb(name,label,frame,labels_folder,path) #get bounding boxes
        else: #if clip contains an action
            sample_bb = []
            clip_info = []
        return sample_bb,clip_info       
    except: #any corrupt files
        print("Exception",file)
        return [],[]

def get_data(annotations_folder, labels_folder,labels_path):
    """
    Input: folder containing annotations, folder containing labels, and labels path
    Output: list with bounding box information of every sample, list with [video names,labels,start_frame,end_frame] of every sample"""
    labels_list = [] #list to store video name, label, and frame
    for file in annotations_folder:
        file_with_labels = get_action_labels(file) #get action labels 
        labels_list = labels_list + file_with_labels 
    labels_list = [labels_list[i:i+3] for i in range(0, len(labels_list), 3)] #each sublist is a sample (video,label,frame)
    all_samples_with_bb = []
    all_clips_info = []
    for sample in labels_list: #obtain bounding boxes for each sample
        name = sample[0].split(".webm")[0]
        label = sample[1]
        frame = int(sample[2])
        sample_with_bb, clip_info = get_bb(name,label,frame,labels_folder,labels_path.split("/*")[0])
        all_samples_with_bb = all_samples_with_bb + [sample_with_bb]
        all_clips_info = all_clips_info + [clip_info] #store clips information so that we can get no-action samples later      
        print(clip_info)       
     
    #get no-action samples
    no_action_bb = [] 
    while len(no_action_bb)<450: 
        file = random.choice(annotations_folder) #random video from the data
        sample_bb,clip_info = get_no_action(file,all_clips_info,labels_folder,labels_path.split("/*")[0]) #obtain its bb information
        if len(sample_bb)!= 0: #if that sequence did not contain an action
            no_action_bb = no_action_bb + [sample_bb]
            all_clips_info = all_clips_info + [clip_info]   
            print(clip_info)  
    data = all_samples_with_bb + no_action_bb #merge action samples with no-action ones
    random.shuffle(data)
    labels = [item[:4] for item in data] #separate clip info (name,label,start_frame,end_frame) from bb data
    for item in data:
        del item[:4]
    return data, labels 



annotations_path = "C:/Users/diego/OneDrive/Documentos/mingle/yolov5/json/new_data2reduced/*"
labels_path = "C:/Users/diego/OneDrive/Documentos/mingle/yolov5/labels/new_data/*"
annotations_folder = glob(annotations_path)
labels_folder = glob(labels_path)
data,labels = get_data(annotations_folder,labels_folder,labels_path) #get all the data ready for the model

#delete any empty sample
index = []
for i in range(len(data)):
    if len(data[i]) == 0:
        index.append(i)
for index in sorted(index, reverse=True):
    del labels[index]
    del data[index]

#pad any samples with less frames
for list in data:
        if len(list)<56:
            for i in range(56-len(list)):
                z1 = [0,0,0,0,0] 
                z2 = [1,0,0,0,0] 
                z3 = [2,0,0,0,0] 
                z4 = [3,0,0,0,0] 
                list.extend([[z1,z2,z3,z4,z4,z4]])

data = np.array([np.array(item) for item in data])
labels = np.array(labels)
#np.save('dataa/new/bb_data.npy', data)
#np.save('dataa/new/labels.npy', labels)
