#!/usr/bin/env python3
"""
Stereo to Mono WAV Converter

This script batch converts stereo WAV files to mono while implementing
strategies to detect and mitigate phase cancellation issues.

Usage:
    python stereo_to_mono_converter.py [--dir DIR] [--method METHOD] [--prefix PREFIX] [--check-phase]

Arguments:
    --dir DIR       Directory containing WAV files (default: current directory)
    --method METHOD Conversion method: avg, left, right, phase_aware (default: phase_aware)
    --prefix PREFIX Prefix for output files (default: "mono_")
    --check-phase   Only check for phase issues without converting
"""

import os
import glob
import argparse
import numpy as np
import soundfile as sf
from tqdm import tqdm
import warnings

def analyze_phase_correlation(left, right):
    """
    Calculate phase correlation between left and right channels.
    Returns a value between -1 and 1:
    - Values close to 1 indicate in-phase (positively correlated)
    - Values close to -1 indicate out-of-phase (negatively correlated)
    - Values close to 0 indicate uncorrelated
    """
    # Normalize and compute correlation
    left_norm = (left - np.mean(left)) / (np.std(left) + 1e-8)
    right_norm = (right - np.mean(right)) / (np.std(right) + 1e-8)
    correlation = np.mean(left_norm * right_norm)
    return correlation

def convert_to_mono(audio_data, method='avg', phase_correlation=None):
    """
    Convert stereo audio to mono using the specified method.
    
    Parameters:
        audio_data (numpy.ndarray): Stereo audio data (shape: [samples, 2])
        method (str): Conversion method:
            - 'avg': Simple average of left and right channels
            - 'left': Use only left channel
            - 'right': Use only right channel
            - 'phase_aware': Adjust based on phase correlation
        phase_correlation (float, optional): Pre-computed phase correlation
        
    Returns:
        numpy.ndarray: Mono audio data (shape: [samples, 1])
    """
    if audio_data.shape[1] != 2:
        raise ValueError("Expected stereo audio (2 channels)")
    
    left = audio_data[:, 0]
    right = audio_data[:, 1]
    
    if method == 'left':
        return left.reshape(-1, 1)
    
    elif method == 'right':
        return right.reshape(-1, 1)
    
    elif method == 'phase_aware':
        # Compute phase correlation if not provided
        if phase_correlation is None:
            phase_correlation = analyze_phase_correlation(left, right)
        
        # If channels are significantly out of phase, adjust before averaging
        if phase_correlation < -0.5:
            print(f"  ⚠️ Phase issue detected (correlation: {phase_correlation:.2f})")
            print("  ⚠️ Inverting right channel phase before summing")
            right = -right  # Invert phase of right channel
            
        # Use weighted average with slight emphasis on the louder channel
        left_energy = np.mean(np.abs(left))
        right_energy = np.mean(np.abs(right))
        total_energy = left_energy + right_energy
        
        if total_energy > 0:
            left_weight = left_energy / total_energy
            right_weight = right_energy / total_energy
            mono = (left * left_weight + right * right_weight)
        else:
            mono = (left + right) / 2
            
        return mono.reshape(-1, 1)
    
    else:  # Default to simple average
        return ((left + right) / 2).reshape(-1, 1)

def process_wav_file(input_file, output_file, method='phase_aware', check_only=False):
    """
    Process a single WAV file from stereo to mono.
    
    Parameters:
        input_file (str): Path to input stereo WAV file
        output_file (str): Path to output mono WAV file
        method (str): Conversion method
        check_only (bool): If True, only check phase without converting
        
    Returns:
        dict: Processing results including phase correlation
    """
    try:
        # Read audio file with SoundFile
        audio_data, sample_rate = sf.read(input_file)
        
        # Check if file is already mono
        if len(audio_data.shape) == 1 or audio_data.shape[1] == 1:
            print(f"  ℹ️ {os.path.basename(input_file)} is already mono. Skipping.")
            return {'filename': input_file, 'already_mono': True}
        
        # Check if file is stereo (2 channels)
        if audio_data.shape[1] != 2:
            print(f"  ⚠️ {os.path.basename(input_file)} has {audio_data.shape[1]} channels, not 2. Skipping.")
            return {'filename': input_file, 'error': f"Expected 2 channels, got {audio_data.shape[1]}"}
        
        # Analyze phase correlation
        left = audio_data[:, 0]
        right = audio_data[:, 1]
        phase_correlation = analyze_phase_correlation(left, right)
        
        result = {
            'filename': input_file,
            'phase_correlation': phase_correlation,
            'phase_issue': phase_correlation < -0.3
        }
        
        # Report phase information
        phase_status = "❌ Potential phase issue" if result['phase_issue'] else "✅ In phase"
        print(f"  Phase correlation: {phase_correlation:.2f} - {phase_status}")
        
        # Convert to mono if not in check-only mode
        if not check_only:
            mono_data = convert_to_mono(audio_data, method, phase_correlation)
            
            # Normalize to prevent clipping
            peak = np.max(np.abs(mono_data))
            if peak > 0.95:
                scaling_factor = 0.95 / peak
                mono_data = mono_data * scaling_factor
                print(f"  ⚠️ Applied gain reduction ({scaling_factor:.2f}) to prevent clipping")
            
            # Write output file
            sf.write(output_file, mono_data, sample_rate)
            result['converted'] = True
            result['output_file'] = output_file
            
            # Report conversion method used
            method_display = {
                'avg': 'Simple average',
                'left': 'Left channel only',
                'right': 'Right channel only',
                'phase_aware': 'Phase-aware mix'
            }.get(method, method)
            print(f"  Method used: {method_display}")
            
        return result
        
    except Exception as e:
        print(f"  ❌ Error processing {input_file}: {str(e)}")
        return {'filename': input_file, 'error': str(e)}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert stereo WAV files to mono with phase cancellation mitigation.')
    parser.add_argument('--dir', type=str, default='.', help='Directory containing WAV files')
    parser.add_argument('--method', type=str, default='phase_aware', 
                        choices=['avg', 'left', 'right', 'phase_aware'],
                        help='Conversion method')
    parser.add_argument('--prefix', type=str, default='mono_', help='Prefix for output files')
    parser.add_argument('--check-phase', action='store_true', help='Only check for phase issues without converting')
    args = parser.parse_args()
    
    # Find all WAV files in the specified directory
    search_pattern = os.path.join(args.dir, '*.wav')
    wav_files = glob.glob(search_pattern)
    wav_files.extend(glob.glob(search_pattern.replace('.wav', '.WAV')))
    
    if not wav_files:
        print(f"No WAV files found in directory: {args.dir}")
        return
    
    print(f"Found {len(wav_files)} WAV files in {args.dir}")
    
    # Process each file
    results = []
    for input_file in tqdm(wav_files, desc="Converting files"):
        print(f"\nProcessing: {os.path.basename(input_file)}")
        
        file_dir, file_name = os.path.split(input_file)
        output_file = os.path.join(file_dir, f"{args.prefix}{file_name}")
        
        result = process_wav_file(
            input_file, 
            output_file, 
            method=args.method,
            check_only=args.check_phase
        )
        results.append(result)
    
    # Summarize results
    total = len(results)
    phase_issues = sum(1 for r in results if r.get('phase_issue', False))
    errors = sum(1 for r in results if 'error' in r)
    already_mono = sum(1 for r in results if r.get('already_mono', False))
    converted = sum(1 for r in results if r.get('converted', False))
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total WAV files: {total}")
    
    if args.check_phase:
        print(f"Files with potential phase issues: {phase_issues} ({phase_issues/total*100:.1f}%)")
    else:
        print(f"Files converted: {converted} ({converted/total*100:.1f}%)")
        print(f"Files already mono: {already_mono} ({already_mono/total*100:.1f}%)")
        print(f"Files with errors: {errors} ({errors/total*100:.1f}%)")
    
    print(f"Files with potential phase issues: {phase_issues} ({phase_issues/total*100:.1f}%)")
    
    if phase_issues > 0:
        print("\nFiles with phase issues:")
        for result in results:
            if result.get('phase_issue', False):
                print(f"  - {os.path.basename(result['filename'])} (correlation: {result['phase_correlation']:.2f})")

if __name__ == "__main__":
    main()
