import requests
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm

import m3u8
import subprocess
from datetime import datetime, timedelta
from polygon_person import detect
import uuid

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from yolov5.utils.general import check_img_size
from typing import List, Tuple
import uvicorn
from pydantic import BaseModel

import torch
import ast
import argparse

file_path = 'log_file.parquet'
result_path = 'result_file.parquet'

app = FastAPI()

class TaskRequest(BaseModel):
    url_param: str
    estimated_duration: int
    polygon_points: List[Tuple[int, int]]


def parse_args_from_dict(arg_dict):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/images', help='source (file/folder, 0 for webcam)')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')
    parser.add_argument('--img-size', type=int, default=1080, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-img', action='store_true', help='save video file to output folder (disable for speed)')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, default=[i for i in range(80)], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    
    parser.add_argument('--sort-max-age', type=int, default=5, help='keep track of object even if object is occluded or not detected in n frames')
    parser.add_argument('--sort-min-hits', type=int, default=2, help='start tracking only after n number of objects detected')
    parser.add_argument('--sort-iou-thresh', type=float, default=0.2, help='intersection-over-union threshold between two frames for association')

    parser.add_argument('--polygon', type=str, help='list of tuples of integers representing a polygon, e.g., "[(589, 431), (713, 429), (732, 478), (603, 482)]"')

    arg_list = []
    for key, value in arg_dict.items():
        if isinstance(value, bool):
            if value:
                arg_list.append(f'--{key}')
        elif isinstance(value, list):
            if key == 'polygon': 
                arg_list.append(f'--{key}')
                arg_list.append(str(value))
            else:
                arg_list.append(f'--{key}')
                arg_list.extend(map(str, value))
        else:
            arg_list.append(f'--{key}')
            arg_list.append(str(value))

    args = parser.parse_args(arg_list)

    if args.polygon:
        args.polygon = ast.literal_eval(args.polygon)
    
    return args

def download_live_video(url_param:str = 'chunklist_w1882981590.m3u8',
                        estimated_duration:int = 30,
                        task_id:str=''):
    
    if estimated_duration > 60:
        estimated_duration = 60

    now_datetime = datetime.now()

    m3u8_url = f"https://cctvjss.jogjakota.go.id/malioboro/Malioboro_25_Utara_Mall.stream/{url_param}"
    stream_url = 'https://cctvjss.jogjakota.go.id/malioboro/Malioboro_25_Utara_Mall.stream/'
    save_directory = 'cctv-temp-download'
    os.makedirs(save_directory, exist_ok=True)
    merge_directory = save_directory + '-merged'
    os.makedirs(merge_directory , exist_ok=True)

    new_data = pd.DataFrame({
        'log_date':[datetime.now()],
        "task_id": [task_id],
        "status": ['downloading video, part: get ts URL']
    })

    if not os.path.exists(file_path):
        new_data.to_parquet(file_path, engine="fastparquet", index=False)
    else:
        new_data.to_parquet(file_path, engine="fastparquet", index=False, append=True)

    ts_url_names = []
    ts_urls = []
    for _ in tqdm(range(estimated_duration)):
        response = requests.get(m3u8_url, verify = False)
        time.sleep(2.25)
        m3u8_content = response.text
        m3u8_obj = m3u8.loads(m3u8_content)
        ts_url_name = [segment.uri for segment in m3u8_obj.segments]
        ts_url = [stream_url + x for x in ts_url_name]
        ts_url_names.extend(ts_url_name)
        ts_urls.extend(ts_url)

    new_data = pd.DataFrame({
        'log_date':[datetime.now()],
        "task_id": [task_id],
        "status": ['downloading video, part: writing ts file']
    })
    new_data.to_parquet(file_path, engine="fastparquet", index=False, append=True)

    for i, (ts_url, ts_name) in tqdm(enumerate(zip(ts_urls, ts_url_names))):
        ts_response = requests.get(ts_url, verify = False)
        ts_filename = os.path.join(save_directory, f'{ts_name}')
        with open(ts_filename, 'wb') as ts_file:
            ts_file.write(ts_response.content)

    ts_files = sorted([os.path.join(save_directory, f) for f in os.listdir(save_directory) if f.endswith('.ts')])

    new_data = pd.DataFrame({
        'log_date':[datetime.now()],
        "task_id": [task_id],
        "status": ['downloading video, part: convert ts to mp4']
    })
    new_data.to_parquet(file_path, engine="fastparquet", index=False, append=True)

    for ts_file in tqdm(ts_files, total = len(ts_files)):
        mp4_file = ts_file.replace('.ts', '.mp4')
        
        subprocess.run([
            'ffmpeg', '-i', ts_file, '-c', 'copy', mp4_file
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        os.remove(ts_file)

    new_data = pd.DataFrame({
        'log_date':[datetime.now()],
        "task_id": [task_id],
        "status": ['downloading video, part: merge mp4 files into one file']
    })
    new_data.to_parquet(file_path, engine="fastparquet", index=False, append=True)

    mp4_files = sorted([os.path.join(save_directory, f) for f in os.listdir(save_directory) if f.endswith('.mp4')])
    first_fl = mp4_files[0].split('/')[-1].replace('.mp4', '')
    last_fl = mp4_files[-1].split('/')[-1].replace('.mp4', '')

    output_video = f"{first_fl}_{last_fl}.mp4"

    clips = [VideoFileClip(file) for file in mp4_files]

    final_clip = concatenate_videoclips(clips, method='compose')
    for mpf in mp4_files:
        os.remove(mpf)
    final_clip.write_videofile(os.path.join(merge_directory, output_video))

    new_data = pd.DataFrame({
        'log_date':[datetime.now()],
        "task_id": [task_id],
        "status": ['downloading video, done']
    })
    new_data.to_parquet(file_path, engine="fastparquet", index=False, append=True)

    return {'video_path' : os.path.join(merge_directory, output_video),
            'video_duration':final_clip.duration,
            'datetime_start':now_datetime,
            'datetime_end':now_datetime + timedelta(seconds=final_clip.duration)}

def calculate_person(vid_file_path:str = 'cctv-temp-download-merged',
                    polygon_points = [(589, 431), (713, 429), (732, 478), (603, 482)],
                    task_id:str=''):

    new_data = pd.DataFrame({
        'log_date':[datetime.now()],
        "task_id": [task_id],
        "status": ['calculate people with yolo']
    })
    new_data.to_parquet(file_path, engine="fastparquet", index=False, append=True)

    args_dict = {
        "weights": "yolov5/weights/yolov5s.pt",
        "source": f"{vid_file_path}",
        "img-size": check_img_size(1080),
        "device": "0",
        "save-img": False,
        "save-txt": False,
        "classes": [0],
        "view-img":False,
        "polygon":polygon_points
    }

    args = parse_args_from_dict(args_dict)
    with torch.no_grad():
        people_passed, total_people_in_frame = detect(args)

    new_data = pd.DataFrame({
        'log_date':[datetime.now()],
        "task_id": [task_id],
        "status": ['calculate people with yolo : done']
    })
    new_data.to_parquet(file_path, engine="fastparquet", index=False, append=True)

    return people_passed, total_people_in_frame


def main_process(url_param:str = 'chunklist_w1882981590.m3u8',
                estimated_duration:int = 30,
                polygon_points:List[Tuple[int, int]] = [(589, 431), (713, 429), (732, 478), (603, 482)],
                task_id:str=''):

    artifacts_download = download_live_video(url_param,
                                            estimated_duration,
                                            task_id)

    
    people_passed, total_people_in_video = calculate_person(artifacts_download['video_path'], polygon_points, task_id)

    os.remove(artifacts_download['video_path'])

    new_data = pd.DataFrame({
        'process_finish_date':[datetime.now()],
        "task_id": [task_id],
        "video_date_start": [artifacts_download['datetime_start']],
        "video_date_end": [artifacts_download['datetime_end']],
        "video_duration":[artifacts_download['video_duration']],
        "people_passed": [people_passed],
        "total_people_in_video":[total_people_in_video]
    })

    if not os.path.exists(result_path):
        new_data.to_parquet(result_path, engine="fastparquet", index=False)
    else:
        new_data.to_parquet(result_path, engine="fastparquet", index=False, append=True)

@app.post("/start-task")
async def start_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
):
    task_id = str(uuid.uuid4())
    background_tasks.add_task(main_process, request.url_param, request.estimated_duration, request.polygon_points, task_id)
    return {"message": "Task started in the background", "task_id":task_id}


@app.post("/task-status")
async def task_status(request: Request):

    df = pd.read_parquet(file_path)

    data = await request.json()
    task_id = data.get("task_id")

    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")

    last_row = df[df["task_id"] == task_id].sort_values('log_date').iloc[-1]

    return {
        "task_id": task_id,
        "status": last_row["status"],
    }


@app.post("/get-result")
async def get_result(request: Request):

    df = pd.read_parquet(result_path)

    data = await request.json()
    task_id = data.get("task_id")

    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")

    last_row = df[df["task_id"] == task_id].iloc[-1]

    return {
        "task_id": task_id,
        "video_date_start": str(last_row["video_date_start"]),
        "video_date_end": str(last_row["video_date_end"]),
        "video_duration":str(last_row['video_duration']),
        "people_passed":str(last_row['people_passed']),
        "total_people_in_video":str(last_row['total_people_in_video']),
        
    }

@app.post("/get-stats")
async def get_result(request: Request):

    df = pd.read_parquet(result_path)
    df['date'] = df['video_date_start'].apply(lambda x: str(x.date()))

    data = await request.json()

    now_date = str(datetime.now().date())

    total_detected_people = df[df['date'] == now_date]['people_passed'].sum()
    last_datetime = str(df.iloc[-1]['video_date_start'])

    last_10_mins = df[df['video_date_end'] > df.iloc[-1]['video_date_start'] - timedelta(minutes=10)]
    last_10_mins_avg = last_10_mins['people_passed'].mean()

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.3, 0.3, 3])


    ax1 = fig.add_subplot(gs[0])
    ax1.text(0.5, 0.5, f"Last datetime video acquired: {last_datetime}", 
            fontsize=12, ha='center', va='center')
    ax1.set_axis_off()

    ax2 = fig.add_subplot(gs[1])
    ax2\
    .text(0.5, 0.5, f"Last 10 minutes AVG: {last_10_mins_avg}\n \n Total people passed polygon {now_date}: {total_detected_people}", 
            fontsize=12, ha='center', va='center')
    ax2.set_axis_off()

    ax3 = fig.add_subplot(gs[2])

    ax3\
    .plot(
        df['video_date_end']\
        .apply(lambda x: str(x)), 
        df['people_passed'], 
        label="People Passed Over Time", 
        marker='o', 
        linestyle='-'
    )

    ax3.set_xlabel("Time")
    ax3.set_ylabel("People Passed")
    ax3.xaxis.set_tick_params(labelsize=5)
    ax3.tick_params(axis='x', rotation=30)
    ax3.legend()
    ax3.grid(True)


    plt.tight_layout()
    plt.savefig("combined_plot.png", dpi=300, format="png")

    with open("combined_plot.png", "rb") as file:
        response = requests.post(
            "https://catbox.moe/user/api.php",
            data={"reqtype": "fileupload"},
            files={"fileToUpload": file}
        )

    time.sleep(3)

    if response.status_code == 200:
        dashboard_url = response.text.strip()
    else:
        dashboard_url = "Fail Upload - " + response.text

    return {
        "last_10_mins_avg": str(last_10_mins_avg),
        "current_date_total_detected_people": str(total_detected_people),
        "latest_video_datetime": str(last_datetime),
        "dashboard_url": dashboard_url
    }


if __name__ == "__main__":
    uvicorn.run("main_app:app", host="127.0.0.1", port=8000, reload=True)