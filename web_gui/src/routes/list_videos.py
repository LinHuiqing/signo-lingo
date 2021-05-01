import os
from flask import render_template

def list_videos():
  try:
    files = os.listdir("./static/uploads")
  except OSError as e:
    print("Error: %s" % (e.strerror))
  return render_template("list_videos.html", files=files)