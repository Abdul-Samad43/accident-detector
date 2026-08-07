import urllib.request

url = "https://ultralytics.com/assets/decelera_landscape_x2.mp4"
urllib.request.urlretrieve(url, "input_video.mp4")
print("Video downloaded!")