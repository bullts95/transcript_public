# Requirements Files

このディレクトリには、プロジェクトで使用する3つの仮想環境それぞれのrequirements.txtファイルが含まれています。

## ファイル一覧

### 1. requirements_whisper.txt
**Whisper環境**（`envs/whisper_env`）用のパッケージリスト

- **目的**: 音声文字起こし（Transcription）
- **主要パッケージ**: 
  - `openai-whisper`: Whisper音声認識モデル
  - ROCm 6.4.1対応PyTorch（`torch`, `torchvision`, `torchaudio`）
- **GPU**: AMD GPU（ROCm）でGPUアクセラレーション

### 2. requirements_pyannote.txt
**Pyannote環境**（`envs/pyannote_env`）用のパッケージリスト

- **目的**: 話者分離（Speaker Diarization）
- **主要パッケージ**:
  - `pyannote-audio`: 話者分離ライブラリ
  - `torch`: PyTorch（CUDA対応）
  - 各種音声処理ライブラリ
- **GPU**: NVIDIA GPU（CUDA）またはCPU

### 3. requirements_app.txt
**Streamlitアプリ環境**（`envs/app_env`）用のパッケージリスト

- **目的**: Webアプリケーションフロントエンド
- **主要パッケージ**:
  - `streamlit`: Webアプリフレームワーク
  - `pandas`: データ処理
- **GPU**: 不要（CPU動作）

## インストール方法

各環境を作成してパッケージをインストールするには：

```bash
# プロジェクトルートディレクトリで実行

# 1. Whisper環境
python3 -m venv envs/whisper_env
source envs/whisper_env/bin/activate
pip install -r requirements/requirements_whisper.txt
deactivate

# 2. Pyannote環境
python3 -m venv envs/pyannote_env
source envs/pyannote_env/bin/activate
pip install -r requirements/requirements_pyannote.txt
deactivate

# 3. Streamlitアプリ環境
python3 -m venv envs/app_env
source envs/app_env/bin/activate
pip install -r requirements/requirements_app.txt
deactivate
```

## 更新方法

環境に新しいパッケージを追加した場合は、以下のコマンドでrequirements.txtを更新してください：

```bash
# Whisper環境の更新
source envs/whisper_env/bin/activate
pip freeze > requirements/requirements_whisper.txt
deactivate

# Pyannote環境の更新
source envs/pyannote_env/bin/activate
pip freeze > requirements/requirements_pyannote.txt
deactivate

# App環境の更新
source envs/app_env/bin/activate
pip freeze > requirements/requirements_app.txt
deactivate
```

## トラブルシューティング

### GPU認識の確認

```bash
# Whisper環境でROCmが認識されているか確認
source envs/whisper_env/bin/activate
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.hip)"
deactivate

# Pyannote環境でCUDAが認識されているか確認
source envs/pyannote_env/bin/activate
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
deactivate
```
