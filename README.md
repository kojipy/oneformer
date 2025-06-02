# OneFormer
OneFormer動作確認用のコード置き場

## setup
```shell
uv sync
```

## 推論
```shell
uv run python src/predict.py data/origin.png
```

## 可視化結果
|入力画像|推論結果|
|---|---|
|![input_image](./data/origin.png)|![segmentation](./data/mask.png)|