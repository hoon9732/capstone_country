import pyautogui, cv2, torch, time, numpy as np
from src.core.pipeline import load_default_pipeline, run_pipeline

model = load_default_pipeline()  # wraps StreetClip + GeoEstimator

print("Left-click-drag to draw ROI, release to classify.")
while True:
    if pyautogui.mouseDown(button="left"):
        x0, y0 = pyautogui.position()
        while pyautogui.mouseDown(button="left"):
            time.sleep(0.01)
        x1, y1 = pyautogui.position()
        box = (min(x0,x1), min(y0,y1), abs(x1-x0), abs(y1-y0))
        grab = pyautogui.screenshot(region=box)
        frame = cv2.cvtColor(np.array(grab), cv2.COLOR_RGB2BGR)
        out = run_pipeline(frame, model)
        print(f"► {out['label']}  ({out['prob']*100:.1f} %)")
