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

## Model information
```bash
python main.py model_info
```
Result:
```
ANN(
  (model): Sequential(
    (0): Linear(in_features=150, out_features=150, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.2, inplace=False)
    (3): Linear(in_features=150, out_features=150, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.2, inplace=False)
    (6): Linear(in_features=150, out_features=150, bias=True)
    (7): ReLU()
    (8): Dropout(p=0.2, inplace=False)
    (9): Linear(in_features=150, out_features=150, bias=True)
    (10): ReLU()
    (11): Dropout(p=0.2, inplace=False)
    (12): Linear(in_features=150, out_features=150, bias=True)
    (13): ReLU()
  )
  (output_layer): Linear(in_features=150, out_features=150, bias=True)
)
Number of parameters: 135900
```
