# YCB Assets Download

```bash
# Create directory
mkdir -p assets/ycb_usd

# Download USD files
curl -o assets/ycb_usd/010_potted_meat_can.usd https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd

curl -o assets/ycb_usd/011_banana.usd https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned/011_banana.usd

curl -o assets/ycb_usd/040_large_marker.usd https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned/040_large_marker.usd

curl -o assets/ycb_usd/005_tomato_soup_can.usd https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd
```

# USD to OBJ Conversion

```bash
blender --background --python usd_to_obj.py
```

> The result files will be generated in `assets/ycb_obj/{*.obj, *.mtl}`.

# BlenderProc Dataset Generation

```bash
blenderproc run generate_dataset.py --num_scenes 10
```

# Convert HDF5 to YOLO Format

```bash
python convert_to_yolo.py
```

# Train YOLO Model

```bash
python train_yolo.py
```