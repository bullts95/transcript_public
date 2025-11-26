import argparse
import os
from pyannote.audio import Pipeline
import torch

def diarize(input_file, num_speakers=None):
    print(f"Loading Pyannote pipeline for {input_file}...")
    
    # Note: You need a valid HF token set in environment or passed to Pipeline
    # Assuming 'config.yaml' or pre-downloaded model, or using use_auth_token=True if logged in
    # For this script, we'll assume standard loading. 
    # If offline, path to model would be needed.
    
    # Check for HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN")
    
    # Try to load from .env if not in environment
    if not hf_token and os.path.exists(".env"):
        print("DEBUG: Loading HF_TOKEN from .env file...")
        try:
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HF_TOKEN="):
                        hf_token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        os.environ["HF_TOKEN"] = hf_token
                        break
        except Exception as e:
            print(f"DEBUG: Error reading .env file: {e}")

    if not hf_token:
        print("DEBUG: HF_TOKEN is NOT set in environment or .env.")
        print("WARNING: You must set HF_TOKEN to access the gated model.")
    else:
        print(f"DEBUG: HF_TOKEN is set (Length: {len(hf_token)}, Starts with: {hf_token[:4]}...)")

    try:
        print("DEBUG: Attempting to load pipeline...")
        # Explicitly pass token if available
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        print("DEBUG: Pipeline loaded successfully.")
        
        # 【重要】パラメータ調整: 短い無音でも話者が変わったと判定させる
        # min_duration_off を0.1s に短縮後、デフォルト(0.5s程度)に戻した 
        print("DEBUG: Optimizing pipeline parameters for short turn-taking...")
        pipeline.instantiate({
            "segmentation": {
                "min_duration_off": 0.2,
                
            }
        })
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: Failed to load Pyannote pipeline.")
        print(f"Error details: {e}")
        
        # Check for GatedRepoError signature in string if we can't import the class easily
        error_str = str(e)
        if "403" in error_str or "GatedRepoError" in error_str or "restricted" in error_str:
            print("\n" + "="*60)
            print("AUTHENTICATION ERROR DETECTED")
            print("="*60)
            print("The model 'pyannote/speaker-diarization-3.1' is a GATED model.")
            print("You must accept the user agreement on Hugging Face to use it.")
            print("\nPlease ensure you have done the following:")
            print("1. Create a Hugging Face account.")
            print("2. Create an Access Token (Read permissions).")
            print("3. Visit https://huggingface.co/pyannote/speaker-diarization-3.1 and accept the license.")
            print("4. Visit https://huggingface.co/pyannote/segmentation-3.0 and accept the license.")
            print("5. Set the HF_TOKEN environment variable or put it in a .env file.")
            print("="*60 + "\n")
        
        import traceback
        traceback.print_exc()
        exit(1)

    # Move to GPU if available
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("Using GPU for diarization.")
    else:
        print("Using CPU for diarization.")

    print("Starting diarization...")
    
    # Run pipeline with num_speakers if provided
    if num_speakers:
        print(f"Running with fixed number of speakers: {num_speakers}")
        diarization = pipeline(input_file, num_speakers=num_speakers)
    else:
        print("Running with automatic speaker detection")
        diarization = pipeline(input_file)

    # Handle DiarizeOutput object (pyannote-audio 3.1+)
    if hasattr(diarization, "speaker_diarization"):
        diarization = diarization.speaker_diarization

    # Save to RTTM
    output_file = input_file.replace(".wav", ".rttm")
    with open(output_file, "w") as f:
        diarization.write_rttm(f)

    print(f"Diarization saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Speaker Diarization using Pyannote")
    parser.add_argument("--input", required=True, help="Path to input WAV file")
    parser.add_argument("--num_speakers", type=int, help="Number of speakers (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found - {args.input}")
        exit(1)

    diarize(args.input, args.num_speakers)
