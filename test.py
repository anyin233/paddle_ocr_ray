import requests
from datasets import load_dataset
import signal
import time
import sys
from concurrent.futures import ThreadPoolExecutor




PROFILER_HOST = "http://localhost"
PROFILER_PORT = 8091

# 读取并编码图片
img_path = 'demo_text_ocr.jpg'

# 构造请求
url = "http://127.0.0.1:8001/ocr"
payload = {"image": open(img_path, 'rb')}
headers = {"Content-Type": "application/json"}

# 发送请求
resp = requests.post(url, files=payload)
print("Status Code:", resp.status_code)
print("OCR Result:", resp.json())


import asyncio

import httpx
from rich.progress import track
import io
import random


# Start profiler
def run_profiler():
  profiler_start_url = f"{PROFILER_HOST}:{PROFILER_PORT}/profiling/start"
  request_body = {
    "name": "ocr_benchmark",
  }
  profiler_start_response = requests.post(profiler_start_url, json=request_body)
  if profiler_start_response.status_code != 200:
      print("Failed to start profiler")
      exit(1)
    
def stop_profiler():
  profiler_stop_url = f"{PROFILER_HOST}:{PROFILER_PORT}/profiling/stop"
  request_body = {
    "name": "ocr_benchmark",
  }
  profiler_stop_response = requests.post(profiler_stop_url, json=request_body)
  if profiler_stop_response.status_code != 200:
      print("Failed to stop profiler")
      exit(1)
      
def submit_ocr_task(url, image_bytes):
  submit_time = time.time()
  response = requests.post(url, files={"image": image_bytes})
  if response.status_code != 200:
    print(f"Error submitting OCR task: {response.status_code} - {response.text}")
    return response, submit_time, None, None
  received_time = response.json().get("received", None)
  finish_time = time.time()
  return response, submit_time, finish_time, received_time
  
async def main():
  responses = []
  
  failed = []
  image_byte_list = []
  ds = load_dataset("getomni-ai/ocr-benchmark", split="test")
  
  
  # Define helper function for processing a single image
  def process_image(index, sample):
    try:
      img_byte_arr = io.BytesIO()
      sample['image'].save(img_byte_arr, format="PNG")
      img_byte_arr.seek(0)
      return (index, img_byte_arr)
    except Exception as e:
      print(f"Error processing image {sample['image']}: {e}")
      return (index, None)

  # Use ThreadPoolExecutor to process images in parallel
  with ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    
    # Submit all tasks
    for index, sample in enumerate(ds):
      futures.append(executor.submit(process_image, index, sample))
    
    # Process results with progress tracking
    for future in track(futures, description="Processing images...", total=len(ds)):
      index, img_bytes = future.result()
      if img_bytes is None:
        failed.append(index)
      else:
        image_byte_list.append(img_bytes)
      
  run_profiler()  
  
  # 随机打乱图片顺序
  random.shuffle(image_byte_list)
  
  # 计算每个请求的发送时间间隔
  total_time = 2.0  # 总时间窗口（秒）
  if len(image_byte_list) > 0:
    interval = total_time / len(image_byte_list)
  else:
    interval = 0
  
  start_time = time.time()
  
  with ThreadPoolExecutor(max_workers=20) as executor:
    futures = []
    submit_timestamp = []
    finish_timestamp = []
    received_timestamp = []
    
    # 在指定时间窗口内提交所有任务
    for index, image_bytes in enumerate(image_byte_list):
      # 计算当前任务应该在什么时候发送
      target_time = start_time + (index * interval)
      current_time = time.time()
      
      # 如果还没到发送时间，等待
      if current_time < target_time:
        time.sleep(target_time - current_time)
      
      futures.append(executor.submit(submit_ocr_task, url,  image_bytes))
      
      
      # Process results with progress tracking
    for future in track(futures, description="Receiving OCR requests...", total=len(image_byte_list)):
      response, submit_time, finish_time, received_time = future.result()
      if response.status_code != 200:
        failed.append(index)
      else:
        responses.append(response.json())
        submit_timestamp.append(submit_time)
        finish_timestamp.append(finish_time)
        received_timestamp.append(received_time)
        
  # for image in track(image_byte_list, description="OCR images...", total=len(image_byte_list)):
  #   payload = {"image": image}
  #   requests.post(url, files=payload)
    # print("Status Code:", response.status_code)
  stop_profiler()
  print("Failed images:", failed)
  with open("timestamp.csv", "w") as f:
    f.write("submit_time,finish_time,received_time\n")
    for submit, finish, receive in zip(submit_timestamp, finish_timestamp, received_timestamp):
      f.write(f"{submit},{finish},{receive}\n")

def handler(signum, frame):
  print("\n收到 SIGINT 信号，执行清理操作...")
  stop_profiler()
  # 在这里做清理工作
  sys.exit(0)
    

if __name__ == "__main__":
 
  # 注册 SIGINT 的处理函数
  signal.signal(signal.SIGINT, handler)
  asyncio.run(main())

