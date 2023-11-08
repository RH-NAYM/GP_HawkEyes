from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from typing import List, Union
from nagad_main_function import *
import uvicorn

app = FastAPI()

class Item(BaseModel):
    url: str
    
async def process_item(item: Item):
    try:
        result = await mainDetect(item.url)
        result = json.loads(result)
        return result
    finally:
        torch.cuda.empty_cache()
        pass

async def process_items(items: Union[Item, List[Item]]):
    print(type(items))
    if type(items)==list:
        coroutines = [process_item(item) for item in items]
        results = await asyncio.gather(*coroutines)
        print("multi : ",results)
    else:
        results = await process_item(items)
        print("single : ", results)
    return results



@app.get("/status")
async def status():
    return "AI Server in running"

@app.post("/nagad")
async def create_items(items: Union[Item, List[Item]]):
    try:
        results = await process_items(items)
        print("Result Sent to User:", results)
        print("###################################################################################################")
        print(items)
        print("Last Execution Time : ", get_bd_time())
        return results
    finally:
        torch.cuda.empty_cache()
        pass

if __name__ == "__main__":
    try:
        del nbrtuModel
        uvicorn.run(app, host="127.0.0.1", port=8000)
    finally:
        torch.cuda.empty_cache()
