# How to reconstruct Vietnamese diacritics

## Data collection, processing & model training
```bash
# make text without diacritics from raw text
nohup python main.py collection > logs/data_collection_0_0_1.log 2>&1 &
# make X, y from text without diacritics and raw text
nohup python main.py processing > logs/data_processing_0_0_1.log 2>&1 &
# train model
nohup python main.py building > logs/model_building_0_0_1.log 2>&1 &
```

## Try to reconstruct Vietnamese diacritics
```bash
python main.py use_app
```

## Model information
```bash
python main.py model_info
```
Result:
```
Transformer(
  (embedding): Embedding(137, 128)
  (encoder_layer): TransformerEncoderLayer(
    (self_attn): MultiheadAttention(
      (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
    )
    (linear1): Linear(in_features=128, out_features=256, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
    (linear2): Linear(in_features=256, out_features=128, bias=True)
    (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
    (dropout1): Dropout(p=0.1, inplace=False)
    (dropout2): Dropout(p=0.1, inplace=False)
  )
  (encoder): TransformerEncoder(
    (layers): ModuleList(
      (0-6): 7 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
        )
        (linear1): Linear(in_features=128, out_features=256, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=256, out_features=128, bias=True)
        (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (output_layer): Linear(in_features=128, out_features=75, bias=True)
)
Number of parameters: 1087051
```
