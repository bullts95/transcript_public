import argparse
import json
import os
import whisper
import torch

def transcribe(input_file, initial_prompt=None):
    print(f"Loading OpenAI Whisper model for {input_file}...")
    
    # Check for ROCm (HIP)
    if torch.cuda.is_available():
        print(f"CUDA (ROCm) is available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("CUDA is NOT available, using CPU.")
        device = "cpu"
    
    model_size = "large-v2" # Stable version
    
    try:
        model = whisper.load_model(model_size, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    print("Starting transcription...")
    
    # Prepare transcription parameters (Strict mode to prevent hallucinations)
    transcribe_params = {
        "language": "ja",
        "beam_size": 5,
        "temperature": 0.0,                # Eliminate randomness
        "condition_on_previous_text": False, # Set "False" to prevent loops from context
        "word_timestamps": True            # Enable word-level timestamps
    }
    
    # Add initial_prompt if provided
    if initial_prompt:
        print(f"Using initial prompt: {initial_prompt}")
        transcribe_params["initial_prompt"] = initial_prompt
    
    result = model.transcribe(input_file, **transcribe_params)

    segments = result["segments"]
    results = []
    for segment in segments:
        print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
        
        # Extract word-level info if available
        words = []
        if "words" in segment:
            words = segment["words"]
            
        results.append({
            "start": segment['start'],
            "end": segment['end'],
            "text": segment['text'],
            "words": words
        })

    # Save to JSON
    output_file = input_file.replace(".wav", ".json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Transcription saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Transcribe audio using OpenAI Whisper")
    parser.add_argument("--input", required=True, help="Path to input WAV file")
    parser.add_argument("--prompt", help="Initial prompt for Whisper (legal terminology, keywords, etc.)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found - {args.input}")
        exit(1)

    transcribe(args.input, args.prompt)
