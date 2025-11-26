import argparse
import json
import pandas as pd
import os

from pyannote.core import Segment, Annotation

def load_rttm_as_annotation(rttm_path):
    """RTTMファイルを読み込んでAnnotationオブジェクトにする"""
    annotation = Annotation()
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9: continue
            start = float(parts[3])
            duration = float(parts[4])
            end = start + duration
            speaker = parts[7]
            annotation[Segment(start, end)] = speaker
    return annotation

def get_dominant_speaker(annotation, start, end):
    """指定区間で最も長く話している話者を返す"""
    segment = Segment(start, end)
    # cropにより、この区間内の話者ラベルとその長さを取得
    cropped = annotation.crop(segment)
    labels = {}
    for seg, _, label in cropped.itertracks(yield_label=True):
        dur = seg.duration
        labels[label] = labels.get(label, 0) + dur
    
    if not labels:
        return "Unknown"
    
    # 最も長い話者を返す
    return max(labels, key=labels.get)

def split_segment_by_speaker(whisper_segment, annotation):
    """
    1つのWhisperセグメント内に明確な話者交代がある場合、分割してリストで返す。
    交代がない場合は、リストに1つだけ入れて返す。
    """
    seg_start = whisper_segment['start']
    seg_end = whisper_segment['end']
    text = whisper_segment['text']
    words = whisper_segment.get('words', [])

    # 単語情報がない場合は分割できないのでそのまま返す
    if not words:
        spk = get_dominant_speaker(annotation, seg_start, seg_end)
        return [{"Start": seg_start, "End": seg_end, "Speaker": spk, "Text": text}]

    # Pyannote上で、このセグメント内に「話者の境界線」があるか探す
    # 判定基準: 0.5秒以上の発言が2つ以上含まれているか？
    segment_annot = annotation.crop(Segment(seg_start, seg_end))
    
    # 簡易的なチェンジポイント検出
    # (話者ラベル, 開始時間) のリストを作成
    changes = []
    for s, _, label in segment_annot.itertracks(yield_label=True):
        # 短すぎるノイズ判定は無視 (例: 0.2秒以下)
        if s.duration > 0.2:
            changes.append((label, s.start, s.end))
    
    # 変化点を時系列順にソート
    changes.sort(key=lambda x: x[1])

    # 明確な話者変更がないなら、そのまま
    unique_speakers = set(c[0] for c in changes)
    if len(unique_speakers) <= 1:
        spk = get_dominant_speaker(annotation, seg_start, seg_end)
        return [{"Start": seg_start, "End": seg_end, "Speaker": spk, "Text": text}]

    # --- 分割ロジック ---
    # 話者が変わるタイミングを見つけて、単語リストを分割する
    
    split_segments = []
    current_words = []
    current_spk = changes[0][0] # 最初の話者
    
    # 2人目以降の話者が始まるタイミングをリスト化
    # changes = [('A', 10.0, 12.0), ('B', 12.1, 15.0)] -> boundaries = [(12.1, 'B')]
    boundaries = []
    for i in range(1, len(changes)):
        prev_spk = changes[i-1][0]
        curr_spk = changes[i][0]
        curr_start = changes[i][1]
        
        if prev_spk != curr_spk:
            boundaries.append((curr_start, curr_spk))
    
    boundary_idx = 0
    
    for word in words:
        w_start = word['start']
        w_end = word['end']
        
        # 次の境界を超えたかチェック
        if boundary_idx < len(boundaries):
            next_boundary_time, next_spk = boundaries[boundary_idx]
            
            # 単語の開始時間が、話者変更点を超えたら分割
            # (少し余裕を持たせるため +0.1秒などの調整もありだが、基本は素直に比較)
            if w_start >= next_boundary_time - 0.1: 
                # これまでのバッファを保存
                if current_words:
                    split_segments.append({
                        "Start": current_words[0]['start'],
                        "End": current_words[-1]['end'],
                        "Speaker": current_spk,
                        "Text": "".join([w['word'] for w in current_words])
                    })
                
                # 新しい話者に切り替え
                current_spk = next_spk
                current_words = []
                boundary_idx += 1
        
        current_words.append(word)
    
    # 残りのバッファを保存
    if current_words:
        split_segments.append({
            "Start": current_words[0]['start'],
            "End": current_words[-1]['end'],
            "Speaker": current_spk,
            "Text": "".join([w['word'] for w in current_words])
        })
        
    return split_segments

def merge_transcription_and_diarization(whisper_result, diarization_result):
    final_data = []
    
    for segment in whisper_result:
        # このセグメントを（必要なら）分割して取得
        processed_segments = split_segment_by_speaker(segment, diarization_result)
        final_data.extend(processed_segments)
            
    return pd.DataFrame(final_data)

def merge_results(wav_path, output_csv=None):
    json_path = wav_path.replace(".wav", ".json")
    rttm_path = wav_path.replace(".wav", ".rttm")
    if output_csv is None:
        output_csv = wav_path.replace(".wav", "_final.csv")

    if not os.path.exists(json_path):
        print(f"Error: JSON transcript not found at {json_path}")
        exit(1)
    if not os.path.exists(rttm_path):
        print(f"Error: RTTM diarization not found at {rttm_path}")
        exit(1)

    print("Loading data...")
    with open(json_path, 'r', encoding='utf-8') as f:
        transcript_segments = json.load(f)
    
    diarization_annotation = load_rttm_as_annotation(rttm_path)

    print("Merging segments (Diarization-Guided Splitting)...")
    df = merge_transcription_and_diarization(transcript_segments, diarization_annotation)
    
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Merged result saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Merge Transcript and Diarization")
    parser.add_argument("--input_wav", help="Path to input WAV file (Single mode)")
    parser.add_argument("--multi_mode", action="store_true", help="Enable multi-channel merge mode")
    parser.add_argument("--input_wavs", nargs="+", help="List of input WAV files (Multi mode)")
    parser.add_argument("--output", help="Path to output CSV file")
    
    args = parser.parse_args()

    if args.multi_mode:
        if not args.input_wavs or not args.output:
            print("Error: --input_wavs and --output are required for multi mode")
            exit(1)
        merge_multi_channel(args.input_wavs, args.output)
    else:
        if not args.input_wav:
            print("Error: --input_wav is required for single mode")
            exit(1)
        merge_results(args.input_wav, args.output)
