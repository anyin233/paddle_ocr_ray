import requests

from paddleocr import draw_ocr

# 读取并编码图片
img_path = 'demo_text_ocr.jpg'

# 构造请求
url = "http://127.0.0.1:8000/ocr"
payload = {"image": open(img_path, 'rb')}
headers = {"Content-Type": "application/json"}

# 发送请求
resp = requests.post(url, files=payload)
print("Status Code:", resp.status_code)
print("OCR Result:", resp.json())

import httpx
import asyncio


async def main():
  responses = []
  
  async with httpx.AsyncClient() as client:
    for _ in range(1):
        payload = {"image": open(img_path, 'rb')}
        async_response = client.post(url, files=payload.copy())
        responses.append(async_response)
    
    for response in responses:
      response = await response
      print("Status Code:", response.status_code)
      print("OCR Result:", response.json())
      

if __name__ == "__main__":
    asyncio.run(main())

# ocr_result = resp.json()
# from PIL import Image
# result = ocr_result[0]
# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path='/home/yanweiye/Project/PaddleOCR/doc/fonts/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save('result.jpg')
