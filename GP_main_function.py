import torch
import asyncio
import json
import pandas as pd
from datetime import datetime
from Data.GP_Data import gpModel
import pytz

def get_bd_time():
    bd_timezone = pytz.timezone("Asia/Dhaka")
    time_now = datetime.now(bd_timezone)
    current_time = time_now.strftime("%I:%M:%S %p")
    return current_time


# Object Detection (Main Function)
async def detect_objects(model, url):
    result = await asyncio.get_event_loop().run_in_executor(None, model, url)
    result = result.pandas().xyxy[0].sort_values(by=['xmin', 'ymax'])
    df = pd.DataFrame(result)
    name_counts = df.groupby('name').size().to_dict()
    result_dict = {}
    for index, row in df.iterrows():
        name = row['name']
        result_dict[name] = name_counts.get(name, 0)
    return result_dict

# Multi-Threading detection 
async def detect_sequence(url):
    gpModel.conf = 0.7
    # tasks = [detect_objects(nbrtuModel, url)]
    # results = await asyncio.gather(*tasks)
    results = await asyncio.create_task(detect_objects(gpModel,url))
    if len(results)==0:
        r = {"object":"no items found"}
    else:
        r = {"object":results}
    nagad_result = json.dumps(r)
    return nagad_result

async def mainDetect(url):
    try:
        result = await detect_sequence(url)
        return result
    finally:
        torch.cuda.empty_cache()
        pass