import streamlit as st
import subprocess
import os
import pandas as pd
import json
import glob
import io
import zipfile
from pathlib import Path

# ==========================================
# 設定：各仮想環境のPythonパス (WSL2環境に合わせて変更してください)
# ==========================================
# プロジェクト内のenvsディレクトリを使用
WHISPER_PYTHON_PATH = os.path.abspath("envs/whisper_env/bin/python")
PYANNOTE_PYTHON_PATH = os.path.abspath("envs/pyannote_env/bin/python")

# スクリプトのパス
SCRIPT_DIR = "scripts"
SCRIPT_TRANSCRIBE = f"{SCRIPT_DIR}/step1_transcribe.py"
SCRIPT_DIARIZE = f"{SCRIPT_DIR}/step2_diarize.py"
SCRIPT_MERGE = f"{SCRIPT_DIR}/step3_merge.py"

# 一時保存フォルダ
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# ==========================================
# ヘルパー関数
# ==========================================
# ==========================================
# ヘルパー関数
# ==========================================
def run_command(command_list, description, env=None):
    """サブプロセスを実行し、エラーがあればStreamlit上で通知する"""
    try:
        # コマンドリストの要素を文字列に変換（念のため）
        cmd_str_list = [str(item) for item in command_list]
        st.write(f"Executing: {' '.join(cmd_str_list)}") # デバッグ用表示
        
        # 環境変数のマージ
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        result = subprocess.run(
            cmd_str_list, 
            check=True, 
            capture_output=True, 
            text=True,
            env=run_env
        )
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"エラーが発生しました: {description}")
        st.error(f"Command: {' '.join(e.cmd)}")
        st.code(e.stderr) # エラー詳細を表示
        return False
    except FileNotFoundError:
        st.error(f"コマンドが見つかりません: {command_list[0]}")
        st.info("Pythonパスやffmpegのインストールを確認してください。")
        return False

def get_audio_info(file_path):
    """ffprobeを使って音声ファイルの情報を取得する"""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        file_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                return {
                    "channels": int(stream.get("channels", 1)),
                    "sample_rate": int(stream.get("sample_rate", 0))
                }
    except Exception as e:
        st.error(f"メタデータ解析エラー: {e}")
    return {"channels": 1, "sample_rate": 0}

def split_channels(input_path, channels):
    """マルチチャネル音声をチャネルごとに分割する"""
    filename = Path(input_path).stem
    output_files = []
    
    for i in range(channels):
        # ch1, ch2, ... (1-based index for filename, 0-based for ffmpeg c0=c{i})
        out_file = os.path.join(TEMP_DIR, f"{filename}_ch{i+1}.wav")
        # pan filter to extract specific channel
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", f"pan=mono|c0=c{i}",
            "-ar", "16000",
            out_file
        ]
        if run_command(cmd, f"チャネル{i+1}抽出"):
            output_files.append(out_file)
            
    return output_files

def build_initial_prompt(user_text: str) -> str:
    """法廷用語プロンプトを生成する"""
    # システム定義（法廷用語の表記揺れ防止）
    system_text = "法廷での口頭弁論および尋問の記録です。過料又は偽証罪を背景に宣誓した上で供述します。なお証拠は、甲１号証、乙２号証、丙３号証、陳述書を含みます。"
    
    # ユーザー入力のサニタイズ（トークン制限対策）
    safe_user_text = user_text.strip()[:50] if user_text else ""
    
    # 結合（ユーザー入力を優先してAttentionを効かせる）
    if safe_user_text:
        return f"{safe_user_text}に関する、{system_text}"
    else:
        return system_text

def convert_and_denoise(input_path):
    """Step 0: ffmpegを使って周波数カット＋ノイズ除去を行い、WAV変換する"""
    # 出力ファイル名
    filename = Path(input_path).stem
    output_path = os.path.join(TEMP_DIR, f"{filename}_clean.wav")
    
    # フィルタチェーンの構築
    # 順序: ハイパス(低音カット) -> ローパス(高音カット) -> ノイズ除去(afftdn)
    # 解説:
    # 1. highpass=f=200: 空調音やマイクの吹かれ音など、200Hz以下の重低音ノイズを物理的にカット。
    # 2. lowpass=f=3000: 3000Hz以上の高音域（キンキンする音やホワイトノイズ）をカット。
    #    ※人の声の主要帯域(300Hz-3000Hz)に絞ることで、PyannoteのVAD誤検知を防ぐ。
    # 3. afftdn=nr=20: 残った帯域内の定常ノイズ（サーッという音）をAIで低減。
    audio_filters = "highpass=f=200,lowpass=f=3000,afftdn=nr=20"

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-af", audio_filters,   # 統合したフィルタを適用
        "-ar", "16000",         # サンプリングレート 16kHz
        "-ac", "1",             # モノラル化
        output_path
    ]
    
    success = run_command(cmd, "Step 0: バンドパス＆ノイズ除去フィルタ適用")
    return output_path if success else None

def format_timestamp(seconds):
    """秒数をHH:MM:SS形式に変換する"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def generate_html_player(df, base_filename, audio_filename):
    """HTMLプレイヤーを生成する"""
    rows = ""
    for _, row in df.iterrows():
        time_formatted = format_timestamp(row['Start'])
        rows += f"<tr><td class='time-col'><span class='timestamp' onclick='seek({row['Start']})'>{time_formatted}</span></td><td class='speaker-col'>{row['Speaker']}</td><td>{row['Text']}</td></tr>"

    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>{base_filename} の文字起こし</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        audio {{ width: 100%; position: sticky; top: 0; background: white; border-bottom: 1px solid #ccc; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; vertical-align: top; }}
        th {{ background-color: #f2f2f2; }}
        .time-col {{ white-space: nowrap; width: 80px; }}
        .speaker-col {{ white-space: nowrap; width: 120px; font-weight: bold; }}
        .timestamp {{ color: #007bff; cursor: pointer; text-decoration: underline; }}
        .timestamp:hover {{ color: #0056b3; }}
    </style>
</head>
<body>
    <h2>{base_filename} の文字起こし</h2>
    <audio id="player" controls src="{audio_filename}"></audio>
    <table>
        <thead><tr><th>時間</th><th>話者</th><th>発言内容</th></tr></thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    <script>
        function seek(seconds) {{
            const player = document.getElementById('player');
            player.currentTime = seconds;
            player.play();
        }}
    </script>
</body>
</html>"""
    return html_content

def generate_summary_text(df):
    """調書用・結合テキストを生成する"""
    text_content = ""
    current_speaker = None
    
    for _, row in df.iterrows():
        speaker = row['Speaker']
        text = row['Text']
        
        if speaker != current_speaker:
            text_content += f"\n【{speaker}】\n{text}"
            current_speaker = speaker
        else:
            text_content += f"{text}"
            
    return text_content.strip()

def generate_raw_text(df):
    """調書用・原文テキストを生成する"""
    return "\n".join(df['Text'].tolist())

def create_output_zip(df, base_filename, audio_filename):
    """
    DataFrameと基本ファイル名から、
    HTML, Summary, Raw, (必要ならCSV) を含むZIPバイナリを作成して返す関数
    """
    base_name = os.path.splitext(base_filename)[0]
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. CSV
        csv_data = df.to_csv(index=False).encode('utf-8-sig')
        zf.writestr(f"{base_name}.csv", csv_data)
        
        # 2. HTML Player
        html_content = generate_html_player(df, base_filename, audio_filename)
        zf.writestr(f"{base_name}_player.html", html_content.encode('utf-8'))
        
        # 3. Summary Text
        summary_content = generate_summary_text(df)
        zf.writestr(f"{base_name}_summary.txt", summary_content.encode('utf-8'))
        
        # 4. Raw Text
        raw_content = generate_raw_text(df)
        zf.writestr(f"{base_name}_raw.txt", raw_content.encode('utf-8'))
        
    return zip_buffer.getvalue(), f"{base_name}_files.zip"

def cleanup_temp_files(file_patterns):
    """一時ファイルを削除する"""
    for pattern in file_patterns:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
                # st.write(f"Deleted: {f}") # Debug
            except Exception as e:
                print(f"Error deleting {f}: {e}")
# ==========================================
# Main GUI
# ==========================================
st.set_page_config(page_title="AI Court Transcriber", layout="wide")
st.title("🎙️ 法廷音声認識システム")
st.markdown("Whisper + Pyannote Separated Pipeline")

# Sidebar for settings
with st.sidebar:
    st.header("設定")
    hf_token = st.text_input("Hugging Face Token", type="password", help="Pyannoteのモデル利用に必要です")
    if not hf_token:
        st.warning("⚠️ Diarizationを実行するにはトークンが必要です")

# --- Input Method Selection ---
tab1, tab2 = st.tabs(["🎙️ 新規書き起こし", "📝 修正CSVから出力作成"])

with tab1:
    input_method = st.radio("入力方法を選択", ["ファイルアップロード", "ローカルフォルダ選択（200MB超のファイルサイズにも対応します。）"])
    
    target_file_path = None
    
    if input_method == "ファイルアップロード":
        uploaded_file = st.file_uploader("音声ファイルをアップロード (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])
        if uploaded_file is not None:
            # 1. ファイルの一時保存
            raw_path = os.path.join(TEMP_DIR, uploaded_file.name)
            with open(raw_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.info(f"ファイルを受け取りました: {uploaded_file.name}")
            target_file_path = raw_path
    
    elif input_method.startswith("ローカルフォルダ選択"):
        folder_path = st.text_input("フォルダパスを入力", value=os.getcwd())
        if os.path.isdir(folder_path):
            files = glob.glob(os.path.join(folder_path, "*"))
            audio_files = [f for f in files if f.lower().endswith(('.mp3', '.wav', '.m4a'))]
            if audio_files:
                selected_filename = st.selectbox("ファイルを選択", [os.path.basename(f) for f in audio_files])
                target_file_path = os.path.join(folder_path, selected_filename)
                st.info(f"選択されたファイル: {target_file_path}")
            else:
                st.warning("音声ファイルが見つかりません。")
        else:
            st.error("無効なフォルダパスです。")
    
    # --- Detailed Settings (Legal Prompt & Options) ---
    user_keywords = ""
    force_stereo_split = False
    
    if target_file_path:
        with st.expander("詳細設定（固有名詞・処理オプション）", expanded=False):
            user_keywords = st.text_input(
                "事件名・人名などの固有キーワード",
                placeholder="例: 令和5年(ワ)第12345号、山田太郎、株式会社A",
                help="認識精度を上げるため、事件固有の名称を入力してください。"
            )
            
            # 話者数の指定UI
            # ファイルごとに設定をリセット/管理するためにkeyを設定
            speaker_count_option = st.selectbox(
                "話者数の指定（ヒントを与える）",
                ["自動判定", "2人（尋問・対談）"] + [f"{i}人" for i in range(3, 11)],
                help="話者数が既知の場合、指定すると精度が向上します。",
                key=f"speaker_count_{target_file_path}"
            )
            
            force_stereo_split = st.checkbox(
                "2チャネル(ステレオ)を強制的に分離して処理する",
                help="ステレオ音源を左右チャネルに分離して個別に処理します。"
            )
    
    # Refactoring UI flow to allow mode selection BEFORE processing
    if target_file_path:
        # Analyze immediately
        audio_info = get_audio_info(target_file_path)
        channels = audio_info["channels"]
        st.write(f"📄 ファイル情報: {os.path.basename(target_file_path)} | チャネル数: {channels}")
    
        process_mode = "mono"
        if channels == 1:
            st.info("ℹ️ モノラル音源: 標準パイプラインで処理します。")
            process_mode = "mono"
        elif channels >= 3:
            st.info("ℹ️ マルチチャネル音源: 各チャネルを分離して個別に処理します。")
            process_mode = "multi"
        elif channels == 2:
            # Check if force_stereo_split checkbox is enabled
            if force_stereo_split:
                st.info("ℹ️ ステレオ音源: 強制分離モードで処理します。")
                process_mode = "multi"
            else:
                stereo_option = st.radio(
                    "ステレオ音源の処理方法を選択:",
                    ["混合して処理 (Diarization使用)", "分離して処理 (L/R分離)"]
                )
                if stereo_option == "分離して処理 (L/R分離)":
                    process_mode = "multi"
                else:
                    process_mode = "mono"
    
        if st.button("処理開始", type="primary"):
            if not hf_token:
                st.error("Hugging Face Tokenを入力してください。")
            else:
                progress_bar = st.progress(0)
                status_area = st.empty()
                
                # Generate initial prompt for Whisper
                initial_prompt = build_initial_prompt(user_keywords)
                st.info(f"🔖 使用プロンプト: {initial_prompt[:100]}..." if len(initial_prompt) > 100 else f"🔖 使用プロンプト: {initial_prompt}")
    
                if process_mode == "mono":
                    # --- Case B: Mono / Stereo Mix ---
                    status_area.text("⏳ Step 0/3: 前処理中 (ノイズ除去 & WAV変換)...")
                    clean_wav_path = convert_and_denoise(target_file_path)
                    
                    if clean_wav_path:
                        progress_bar.progress(20)
                        
                        # Step 1: Transcribe
                        status_area.text("⏳ Step 1/3: 文字起こしを実行中 (Whisper - GPU)...")
                        cmd_transcribe = [
                            WHISPER_PYTHON_PATH, SCRIPT_TRANSCRIBE,
                            "--input", clean_wav_path,
                            "--prompt", initial_prompt
                        ]
                        if run_command(cmd_transcribe, "Whisper文字起こし"):
                            progress_bar.progress(50)
                            
                            # Step 2: Diarize
                            status_area.text("⏳ Step 2/3: 話者分離を実行中 (Pyannote)...")
                            cmd_diarize = [
                                PYANNOTE_PYTHON_PATH, SCRIPT_DIARIZE,
                                "--input", clean_wav_path
                            ]
                            
                            # 話者数が指定されている場合、引数を追加
                            if speaker_count_option != "自動判定":
                                # "2人（尋問・対談）" -> 2, "10人" -> 10
                                # "人"で分割して数値を取り出す
                                num_str = speaker_count_option.split("人")[0]
                                num_speakers = int(num_str)
                                cmd_diarize.extend(["--num_speakers", str(num_speakers)])
    
                            env_vars = {"HF_TOKEN": hf_token}
                            
                            if run_command(cmd_diarize, "Pyannote話者分離", env=env_vars):
                                progress_bar.progress(80)
                                
                                # Step 3: Merge
                                status_area.text("⏳ Step 3/3: データを統合中...")
                                cmd_merge = [
                                    PYANNOTE_PYTHON_PATH, SCRIPT_MERGE,
                                    "--input_wav", clean_wav_path
                                ]
                                if run_command(cmd_merge, "データ統合"):
                                    progress_bar.progress(100)
                                    status_area.success("✅ 完了しました！")
                                    
                                    # Result
                                    csv_path = clean_wav_path.replace(".wav", "_final.csv")
                                    if os.path.exists(csv_path):
                                        df = pd.read_csv(csv_path)
                                        st.subheader("📝 解析結果")
                                        st.dataframe(df.head(10))
                                        
                                        # ZIP Download
                                        zip_data, zip_name = create_output_zip(df, os.path.basename(target_file_path), os.path.basename(target_file_path))
                                        st.download_button(
                                            label="📥 結果ファイルを一括ダウンロード (ZIP)",
                                            data=zip_data,
                                            file_name=zip_name,
                                            mime="application/zip",
                                            help="ダウンロード後、サーバー上の一時ファイルはすべて削除されます。"
                                        )
                                        
                                        # Cleanup
                                        cleanup_files = [
                                            clean_wav_path,
                                            clean_wav_path.replace(".wav", ".json"),
                                            clean_wav_path.replace(".wav", ".rttm"),
                                            csv_path
                                        ]
                                        # Also delete uploaded file if it's in temp
                                        if target_file_path.startswith(TEMP_DIR):
                                            cleanup_files.append(target_file_path)
                                            
                                        cleanup_temp_files(cleanup_files)
                                        status_area.info("🗑️ 一時ファイルを削除しました。")
    
                elif process_mode == "multi":
                    # --- Case A: Multi-channel Split ---
                    status_area.text("⏳ Step 0/3: チャネル分割中...")
                    
                    # 1. Split Channels
                    channel_files = split_channels(target_file_path, channels)
                    if not channel_files:
                        st.error("チャネル分割に失敗しました。")
                    else:
                        progress_bar.progress(20)
                        
                        # 2. Transcribe Each Channel
                        status_area.text("⏳ Step 1/2: 各チャネルを文字起こし中 (Whisper)...")
                        
                        # Process sequentially (could be parallelized but GPU VRAM might be an issue)
                        for i, ch_file in enumerate(channel_files):
                            status_area.text(f"⏳ 文字起こし中: Channel {i+1}/{len(channel_files)}...")
                            cmd_transcribe = [
                                WHISPER_PYTHON_PATH, SCRIPT_TRANSCRIBE,
                                "--input", ch_file,
                                "--prompt", initial_prompt
                            ]
                            if not run_command(cmd_transcribe, f"Whisper (Ch {i+1})"):
                                st.error(f"Channel {i+1} の処理中にエラーが発生しました。")
                                break
                        else:
                            # All channels processed successfully
                            progress_bar.progress(70)
                            
                            # 3. Merge (New Logic)
                            status_area.text("⏳ Step 2/2: 全チャネルのデータを統合中...")
                            
                            # We will use the first channel file as the "primary" input for the script to locate others?
                            # Or just pass the list of wavs (which have associated jsons).
                            
                            # Let's construct a command that passes all wav files
                            cmd_merge = [
                                PYANNOTE_PYTHON_PATH, SCRIPT_MERGE,
                                "--multi_mode",
                                "--input_wavs"
                            ] + channel_files
                            
                            # We need to define the output path.
                            output_csv = os.path.join(TEMP_DIR, f"{Path(target_file_path).stem}_final.csv")
                            
                            cmd_merge.extend(["--output", output_csv])
                            
                            if run_command(cmd_merge, "データ統合 (Multi)"):
                                progress_bar.progress(100)
                                status_area.success("✅ 完了しました！")
                                
                                if os.path.exists(output_csv):
                                    df = pd.read_csv(output_csv)
                                    st.subheader("📝 解析結果 (マルチチャネル統合)")
                                    st.dataframe(df.head(10))
                                    
                                    # ZIP Download
                                    zip_data, zip_name = create_output_zip(df, os.path.basename(target_file_path), os.path.basename(target_file_path))
                                    st.download_button(
                                        label="📥 結果ファイルを一括ダウンロード (ZIP)",
                                        data=zip_data,
                                        file_name=zip_name,
                                        mime="application/zip",
                                        help="ダウンロード後、サーバー上の一時ファイルはすべて削除されます。"
                                    )
                                    
                                    # Cleanup
                                    cleanup_files = [output_csv]
                                    for ch_file in channel_files:
                                        cleanup_files.append(ch_file)
                                        cleanup_files.append(ch_file.replace(".wav", ".json"))
                                        
                                    # Also delete uploaded file if it's in temp
                                    if target_file_path.startswith(TEMP_DIR):
                                        cleanup_files.append(target_file_path)
    
                                    cleanup_temp_files(cleanup_files)
                                    status_area.info("🗑️ 一時ファイルを削除しました。")
                                else:
                                    st.error("結果ファイルが生成されませんでした。")

with tab2:
    st.header("修正済みCSVから各フォーマットを生成")
    st.markdown("一度ダウンロードしたCSVをExcel等で修正し、ここにアップロードすることで、HTMLやテキストファイルを再生成できます。")
    
    uploaded_csv = st.file_uploader("CSVファイルをアップロード", type=["csv"])
    audio_name = st.text_input("音声ファイル名（拡張子含む）", placeholder="example.mp3", help="HTMLプレイヤーが参照するファイル名です。")
    
    if uploaded_csv and audio_name:
        if st.button("フォーマット変換・ZIP作成"):
            try:
                # CSV読み込み
                df_fixed = pd.read_csv(uploaded_csv)
                
                # 必須カラムチェック
                required_cols = ["Start", "End", "Speaker", "Text"]
                if not all(col in df_fixed.columns for col in required_cols):
                    st.error(f"CSVの形式が正しくありません。以下の列が必要です: {required_cols}")
                else:
                    # 共通関数でZIP生成
                    # base_filenameはCSVファイル名から拡張子を除いたもの
                    base_name = os.path.splitext(uploaded_csv.name)[0]
                    zip_data, zip_name = create_output_zip(df_fixed, base_name, audio_name)
                    
                    st.success("生成完了！")
                    st.download_button(
                        label="📦 修正版ZIPをダウンロード",
                        data=zip_data,
                        file_name=zip_name,
                        mime="application/zip"
                    )
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
