# How to reconstruct Vietnamese diacritics

## Data processing & model training
```bash
nohup python main.py processing > logs/data_processing.log 2>&1 &
# about 7m
nohup python main.py building > logs/model_building.log 2>&1 &
```

## Try to reconstruct Vietnamese diacritics
```bash
# nohup python main.py use_app > logs/app.log 2>&1 &
python main.py use_app
```
