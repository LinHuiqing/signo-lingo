import os
import glob
from flask import redirect

def clear_videos():
  files = glob.glob("./static/uploads/**/*.mp4", recursive=True)
  for f in files:
    try:
      os.remove(f)
    except OSError as e:
      print("Error: %s : %s" % (f, e.strerror))
  return redirect("/")