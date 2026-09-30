"""Example model adapter (random guesses) - copy this and plug in your real classifier.

Contract:
  load(weights_path) -> model          optional; called once
  predict(model, image_paths) -> list  one dict per path:
      {"color": "red" | None,          # tag id or name (case-insensitive) from the Tag palette
       "color_conf": 0.0-1.0,
       "number": 1-100 | None,
       "number_conf": 0.0-1.0}
  (predict(image_paths) is also accepted if you have no load())
"""
import random

COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "white"]


def load(weights_path):
    return {"weights": weights_path}


def predict(model, image_paths):
    return [{"color": random.choice(COLORS), "color_conf": random.random(),
             "number": random.randint(1, 100), "number_conf": random.random()} for _ in image_paths]
