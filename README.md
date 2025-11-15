```bash
blender --background --python usd_to_obj.py
```

```bash
blenderproc run generate_dataset.py --num_scenes 10
```

```bash
python convert_to_yolo.py
```

```bash
python train_yolo.py
```