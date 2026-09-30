# Bee Tagger

Standalone (Python stdlib only, no Remi files needed):

    python3 server.py --port 8002      # then open http://localhost:8002
    # remote: ssh -L 8002:localhost:8002 <host>

Click **Open CSV…** and type/browse to a tracks CSV. Images load from its `crop_filepath` column
(fallback: `<csv folder>/crops/<crop_filename>`). Source CSVs are never modified.
Labels autosave to `labels/<dataset>.json`; **Export CSV** writes `labels/<dataset>.tagged.csv` with
`track_key_sam, tag_color(+_source), tag_number(+_source), tag_rotation, pred_*` columns (rows added by Merge included).

## Tabs
- **Tag** – color buttons + numbers 1–100, per track. Select tracks (click / shift / ctrl), then a color button
  (default hotkeys `Q W E R…`; right-click a color button → **Hotkey** to rebind), or type digits + Enter (or number box / Grid). Double-click opens a track; inside, buttons tag the
  whole track and right-click an image tags only that image (color and/or number).
  Right-click a color button to rename/recolor/reorder/delete. Ctrl+Z undoes.
- **Inference** – run your model, then review: sort by lowest confidence, filter "needs review",
  `A` accepts the prediction, or fix by hand with the same buttons.
- **Merge** – pick another CSV; rows with the same `crop_filepath` get their color / number / rotation copied in
  (source column names are auto-detected, e.g. `Tag_color`, `ground_truth_numbers`, `tag_rotation`; type the name if not
  found). Optionally add rows not in the current CSV (their `track_id`s are renumbered to be unique per video; they are stored
  in `labels/<dataset>.extra.csv`, the source CSV is untouched). "Undo last merge" restores the pre-merge state.
- **Rotation** – every image that has a color + number, one at a time. `A`/`S` rotate left/right by the step (default 15°),
  `Z`/`X` or Ctrl+A / Ctrl+S / Ctrl+click = 1°, Enter saves and goes to the next. Saved as `tag_rotation` (degrees clockwise)
  in the export.
- **Distribution** – tagged totals, color bars, number histogram stacked by color; click bars / set a number
  range to cross-filter; toggle Images vs Tracks.

## Plugging in your model
Point the Inference tab at a `.py` file (plus optional weights and the Python of your training env):

    def load(weights_path): ...                    # optional
    def predict(model, image_paths):               # -> one dict per path
        return [{"color": "red", "color_conf": .9, "number": 42, "number_conf": .8}, ...]

`color` is a tag id or name from the palette; `number` is 1–100. See `models/example_model.py`.
