from pathlib import Path
import numpy as np
for ppm in Path("camera_snaps").glob("*.ppm"):
    with open(ppm, "rb") as f:
        header = f.readline(); size = f.readline(); _ = f.readline()
        w,h = map(int, size.split())
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(h, w, 3)
    from PIL import Image  # install pillow if not present: pip install pillow
    Image.fromarray(data).save(ppm.with_suffix(".png"))
