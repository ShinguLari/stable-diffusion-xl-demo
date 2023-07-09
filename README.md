---
title: Stable Diffusion XL 0.9
emoji: 🔥
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 3.11.0
app_file: app.py
pinned: true
license: mit
---

# StableDiffusion XL Gradio Demo
これはGradioのデモで [Stable Diffusion XL 0.9](https://github.com/Stability-AI/generative-models)をサポートします.このデモでは、BASEとRefinerのモデルをロードします。

This is forked from [StableDiffusion v2.1 Demo](https://huggingface.co/spaces/gradio-client-demos/stable-diffusion). Refer to the git commits to see the changes.

Update:Google Colabに対応しています。T4でもColab上でこのデモを無料で実行できます。 <a target="_blank" href="https://colab.research.google.com/github/TonyLianLong/stable-diffusion-xl-demo/blob/main/Stable_Diffusion_XL_Demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## 作例

左: SDXL 0.9. 右: [SD v2.1](https://huggingface.co/spaces/gradio-client-demos/stable-diffusion).

チューニング無しで、SDXLはSD v2.1に比べてはるかに優れた画像を生成します！

### 作例 1
<p align="middle">
<img src="imgs/img1_sdxl0.9.png" width="48%">
<img src="imgs/img1_sdv2.1.png" width="48%">
</p>

### 作例 2
<p align="middle">
<img src="imgs/img2_sdxl0.9.png" width="48%">
<img src="imgs/img2_sdv2.1.png" width="48%">
</p>

### 作例 3
<p align="middle">
<img src="imgs/img3_sdxl0.9.png" width="48%">
<img src="imgs/img3_sdv2.1.png" width="48%">
</p>

### 作例 4
<p align="middle">
<img src="imgs/img4_sdxl0.9.png" width="48%">
<img src="imgs/img4_sdv2.1.png" width="48%">
</p>

### 作例 5
<p align="middle">
<img src="imgs/img5_sdxl0.9.png" width="48%">
<img src="imgs/img5_sdv2.1.png" width="48%">
</p>

## インストール
torch 2.0.1がインストールされている場合、次のファイルもインストールする必要があります。:
```shell
pip install accelerate transformers invisible-watermark "numpy>=1.17" "PyWavelets>=1.1.1" "opencv-python>=4.1.0.25" safetensors "gradio==3.11.0"
pip install git+https://github.com/huggingface/diffusers.git@sd_xl
```

## 起動
無料ですが、ウェイトにアクセスする必要がある。 [submit a quick form](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9) 
ウェイトをロードするには2つの方法がある。ウェイトにアクセスした後、ローカルにcloneするか、このrepoでウェイトをロードするかです。


### オプション 1
もし両方のreopをローカルでcloneした場合 ([base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9), [refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-0.9))  (以下に変えて下さい `path_to_sdxl`):
```
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 SDXL_MODEL_DIR=/path_to_sdxl python app.py
```

### オプション 2
huggingface hub からロードしたい場合は(以下をセットアップしてください [HuggingFace access token](https://huggingface.co/docs/hub/security-tokens)):
```
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 ACCESS_TOKEN=YOUR_HF_ACCESS_TOKEN python app.py
```

### `torch.compile` サポート
`torch.compile`をオンにすると、全体的な推論が速くなります。しかしこれは最初の実行に若干のオーバーヘッドを追加することになります（つまり、最初の実行時にコンパイルを待つ必要があります）。

### メモリを節約する
1. app.py`の`pipe.enable_model_cpu_offload()`をオンにし、`pipe.to("cuda")`をオフにする。
2. enable_refiner`をFalseにしてRefinerをオフにする。
3. 以下の方法を更に知るには [メモリを節約し、物事をより速くする](https://huggingface.co/docs/diffusers/optimization/fp16).

### 環境変数によるいくつかのオプション
* SDXL_MODEL_DIR` と `ACCESS_TOKEN`: ローカルまたは HF hub から SDXL をロードする。
* ENABLE_REFINER=true/false` Refinerをオン/オフする。([refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-0.9) 世代を絞り込む).
* `OUTPUT_IMAGES_BEFORE_REFINER=true/false` リファイナーが有効な場合に有効。リファイナーステージ前後の画像を出力する。
* `SHARE=true/false` 公開リンクを作成する（共有やコラボに便利）

## If you enjoy this demo, please give [this repo](https://github.com/TonyLianLong/stable-diffusion-xl-demo) a star ⭐.
