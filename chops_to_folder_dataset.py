# Copyright (C) 2023 by Artem Khrapov (kabachuha)
# Read LICENSE for usage terms.

import os
import sys
import argparse
import cv2
from tqdm import tqdm
import shutil
from pathlib import Path
from PIL import Image
import time, logging

def write_as_video(output_filename, video_frames, overwrite_dims, width, height, fps):
    
    if not overwrite_dims:
        height, width, _ = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    for j in video_frames:
       j = cv2.resize(j, (width, height), interpolation= cv2.INTER_CUBIC)
       out.write(j)
    # video_frames = [Image.fromarray(cv2.cvtColor(j, cv2.COLOR_BGR2RGB)) for j in video_frames]
    # if overwrite_dims:
    #     video_frames = [j.resize(width, height) for j in video_frames]

    # video_frames[0].save(output_filename, save_all=True, append_images=video_frames[1:])

    out.release()

def read_first_frame(video_path):
    patience = 5
    p = 0
    video = cv2.VideoCapture(video_path)
    ret = False
    while not ret:
        ret, frame = video.read()
        p += 1
        if p > patience:
            raise Exception(f'Cannot read video at {video_path}')
    video.release()
    return frame

def read_all_frames(video_path):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frames = []

    for _ in range(total_frames):
        ret, frame = video.read()
        video_frames.append(frame)
    video.release()
    return video_frames

def get_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    video.release()
    return fps

def calculate_depth(init_path):
    max_depth = 20
    depth_name = init_path

    for d in range(0, max_depth):
        depth_name = os.path.join(depth_name, f'depth_{d}')
        if not os.path.exists(depth_name):
            # count all subsets in a first full part
            L = len(os.listdir(os.path.join(init_path, 'depth_0', 'depth_1', 'depth_2', 'part_0'))) // 2
            return d, L

def move_the_files(init_path, L, depth, overwrite_dims, width, height, overwrite_fps, fps):

    folder_dataset_path = os.path.join(init_path, 'folder_dataset')
    os.mkdir(folder_dataset_path)
    depth_name = init_path

    t_counter=0
    for d in range(0, depth+1):
        for j in range(L**(d-1) if d > 1 else 1):
            for i in range(L if d > 0 else 1):
                t_counter+=1
    tq = tqdm(total=t_counter)

    for d in range(0, depth):
        depth_name = os.path.join(depth_name, f'depth_{d}')
        for j in range(L**(d-1) if d > 1 else 1):
            part_path = os.path.join(depth_name, f'part_{j}')
                # sample the text info for the next subset
            for i in range(L if d > 0 else 1):
                txt_path = os.path.join(part_path, f'subset_{i}.txt')
                
                # go to the subset for video frames sampling
                next_depth_name = os.path.join(depth_name, f'depth_{d+1}')
                next_part_path = os.path.join(next_depth_name, f'part_{i+L*j}') # `i` cause we want to sample each corresponding *subset*

                # depths > 0 are *guaranteed* to have L videos in their part_j folders
                
                # now sampling each first frame at the next level
                L_frames = [read_first_frame(os.path.join(next_part_path, f'subset_{k}.mp4')) for k in range(L)]
                
                # write all the L sampled frames to an mp4 in the folder dataset
                if overwrite_fps:
                    fps = get_fps(os.path.join(next_part_path, f'subset_{0}.mp4'))
                
                write_as_video(os.path.join(folder_dataset_path, f'depth_{d}_part_{j}_subset{i+L*j}.mp4'), L_frames, overwrite_dims, width, height, fps)
                shutil.copy(txt_path, os.path.join(folder_dataset_path, f'depth_{d}_part_{j}_subset{i+L*j}.txt'))

                tq.set_description(f'Depth {d}, part {j}, subset{i}')
                #tq.set_description(os.path.join(next_part_path, f'subset_{0}.mp4'))
                tq.update(1)

    # collecting the deepest level L-frame long mp4s as is
    d = depth
    depth_name = os.path.join(depth_name, f'depth_{d}')
    for j in range(L**(d-1) if d > 1 else 1):
        part_path = os.path.join(depth_name, f'part_{j}')
        for i in range(L if d > 0 else 1):
            txt_path = os.path.join(part_path, f'subset_{i}.txt')
            mp4_path = os.path.join(part_path, f'subset_{i}.mp4')

            write_as_video(os.path.join(folder_dataset_path, f'depth_{d}_part_{j}_subset{i+L*j}.mp4'), read_all_frames(mp4_path), overwrite_dims, width, height, fps)
            shutil.copy(txt_path, os.path.join(folder_dataset_path, f'depth_{d}_part_{j}_subset{i+L*j}.txt'))
            tq.set_description(f'Depth {d}, part {j}, subset{i}')
            tq.update(1)
    tq.close() 

def main():
    parser = argparse.ArgumentParser(description="Convert the chopped labeled tree-like data into a FolderDataset")
    parser.add_argument("outpath", help="Path where to save the end FolderDataset", default=os.getcwd())
    parser.add_argument("--L", help="Num of splits on each level.")
    parser.add_argument("--D", help="Tree depth")
    parser.add_argument("--overwrite_dims", help="Preserve the original video dims", action="store_true")
    parser.add_argument("--w", help="Output video width", default=384)
    parser.add_argument("--h", help="Output video height", default=256)
    parser.add_argument("--overwrite_fps", help="Preserve the original video fps", action="store_true")
    parser.add_argument("--fps", help="Output video fps", default=12)
    args = parser.parse_args()
    move_the_files(args.outpath, int(args.L), int(args.D), bool(args.overwrite_dims), int(args.w), int(args.h), bool(args.overwrite_fps), int(args.fps))

if __name__ == "__main__":
    main()
    