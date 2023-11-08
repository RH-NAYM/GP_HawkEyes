import torch
import asyncio
import json
import pandas as pd
from datetime import datetime
from Data.NBRTU_Data import nbrtuModel, NBRTU_val, ndel_items, nagad_items, bkash_items, rocket_items, tap_items, upay_items
import pytz

def get_bd_time():
    bd_timezone = pytz.timezone("Asia/Dhaka")
    time_now = datetime.now(bd_timezone)
    current_time = time_now.strftime("%I:%M:%S %p")
    return current_time

# Nagad Split Function
async def process_nagad_item(nagad_item, nbrtuDict, nagad):
    if nagad_item in nbrtuDict:
        n = {nagad_item: nbrtuDict[nagad_item]}
        nagad.update(n)

# Bkash Split Function
async def process_bkash_item(bkash_item, nbrtuDict, bkash):
    if bkash_item in nbrtuDict:
        b = {bkash_item: nbrtuDict[bkash_item]}
        bkash.update(b)

# Rocket Split Function
async def process_rocket_item(rocket_item, nbrtuDict, rocket):
    if rocket_item in nbrtuDict:
        r = {rocket_item: nbrtuDict[rocket_item]}
        rocket.update(r)

# Tap Split Function
async def process_tap_item(tap_item, nbrtuDict, tap):
    if tap_item in nbrtuDict:
        t = {tap_item: nbrtuDict[tap_item]}
        tap.update(t)

# Upay Split Function
async def process_upay_item(upay_item, nbrtuDict, upay):
    if upay_item in nbrtuDict:
        u = {upay_item: nbrtuDict[upay_item]}
        upay.update(u)


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
    nbrtuModel.conf = 0.7
    # tasks = [detect_objects(nbrtuModel, url)]
    # results = await asyncio.gather(*tasks)
    results = await asyncio.create_task(detect_objects(nbrtuModel,url))
    nbrtuDict = results

    # Nagad Bkash Rocket Tap Upay Validation :
    for val_item in NBRTU_val:
        if val_item in nbrtuDict and nbrtuDict[val_item] > 0:
            nbrtu_validation_single = {val_item: "yes"}
            nbrtuDict.update(nbrtu_validation_single)


    # Remove Extra Items : 
    for nagad_remove_item in ndel_items:
        if nagad_remove_item in nbrtuDict:
            del nbrtuDict[nagad_remove_item]

    nagad = {}
    bkash = {}
    rocket = {}
    tap = {}
    upay = {}

    # Using asyncio.gather to await multiple process functions concurrently
    process_nagad_tasks = [process_nagad_item(nagad_item, nbrtuDict, nagad) for nagad_item in nagad_items]
    process_bkash_tasks = [process_bkash_item(bkash_item, nbrtuDict, bkash) for bkash_item in bkash_items]
    process_rocket_tasks = [process_rocket_item(rocket_item, nbrtuDict, rocket) for rocket_item in rocket_items]
    process_tap_tasks = [process_tap_item(tap_item, nbrtuDict, tap) for tap_item in tap_items]
    process_upay_tasks = [process_upay_item(upay_item, nbrtuDict, upay) for upay_item in upay_items]

    await asyncio.gather(*process_nagad_tasks, *process_bkash_tasks, *process_rocket_tasks, *process_tap_tasks, *process_upay_tasks)

    nagad_detection = {
        'nagad': nagad,
        'bkash': bkash,
        'rocket': rocket,
        'tap': tap,
        'upay': upay
    }


    nagad_result = json.dumps(nagad_detection)
    return nagad_result

async def mainDetect(url):
    try:
        result = await detect_sequence(url)
        return result
    finally:
        torch.cuda.empty_cache()
        pass