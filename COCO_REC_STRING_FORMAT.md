# COCO Format With `rec_string` (SPTSv2)

This project uses a COCO-style JSON with text-recognition fields.

## Top-Level Structure

The JSON must contain:

- `images`: list of image metadata
- `annotations`: list of text instances
- `categories`: usually one class (`text`)

```json
{
  "images": [],
  "annotations": [],
  "categories": [
    {"id": 1, "name": "text"}
  ]
}
```

## `images` Entry

Each image object:

- `id` (int): unique image id
- `file_name` (str): image filename relative to image folder
- `width` (int): image width in pixels
- `height` (int): image height in pixels

Example:

```json
{
  "id": 1,
  "file_name": "ICDAR2019_tr_img_00001.jpg",
  "width": 1280,
  "height": 720
}
```

## `annotations` Entry

Each annotation object:

- `id` (int): unique annotation id
- `image_id` (int): id of the image this text belongs to
- `category_id` (int): class id (`1` for text)
- `bbox` (list[float], len=4): `[x, y, w, h]`
- `area` (float): usually `w * h`
- `iscrowd` (int): usually `0`
- `bezier_pts` (list[float], len=16): 8 control points as
  `[x1, y1, x2, y2, ..., x8, y8]`
- `rec` (list[int], fixed len): encoded text sequence (padded)
- `rec_string` (str): original/decoded text string

Example:

```json
{
  "id": 25,
  "image_id": 1,
  "category_id": 1,
  "bbox": [104.0, 212.0, 183.0, 46.0],
  "area": 8418.0,
  "iscrowd": 0,
  "bezier_pts": [104.0, 212.0, 165.0, 212.0, 226.0, 212.0, 287.0, 212.0, 287.0, 258.0, 226.0, 258.0, 165.0, 258.0, 104.0, 258.0],
  "rec": [45, 50, 38, 52, 14, 31, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96],
  "rec_string": "TEXT-1"
}
```

## `categories` Entry

Typical setup:

```json
[
  {"id": 1, "name": "text"}
]
```

Some scripts may include:

```json
[
  {"id": 1, "name": "text", "supercategory": "text"}
]
```

Both are acceptable for this project.

## Notes

- `bezier_pts` can be integer or real values in JSON; the training pipeline reads them as float tensors.
- Keep `rec` and `rec_string` consistent whenever possible.
- During merges, image filenames are often prefixed by dataset name:
  `"{dataset_name}_{original_file_name}"`.
