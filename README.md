## Development Stage
Complete

## stereo-to-mono basic information
- Copies and converts an entire folder of stereo audio files into new mono files. Audio phase issues are mitigated by functions within the code.

## Install required dependancies
- python
- pip install numpy soundfile tqdm

## Run the program
- place stereo_to_mono_converter.py in the folder with the stereo files that you want to be mono
- execute "python stereo_to_mono_converter.py" in the command prompt (that has been directed to the same folder)

## Advanced options
- Specify a different directory
  - python stereo_to_mono_converter.py --dir /path/to/your/samples
- Use a different conversion method
  - python stereo_to_mono_converter.py --method left
- Change the prefix for output files (default is "mono_")
  - python stereo_to_mono_converter.py --prefix "m_"
- Only check for phase issues without converting
  - python stereo_to_mono_converter.py --check-phase
  
## Phase Cancellation Detection & Mitigation
- The core of the program analyzes the correlation between left and right channels. When it detects negatively correlated audio (indicating potential phase issues), it:
  - Automatically identifies problematic files
  - Inverts the phase of one channel before summing when necessary
  - Applies intelligent channel weighting to preserve maximum audio information

## Multiple Conversion Methods
### The program supports four different conversion strategies:
  - phase_aware (default): Intelligently handles phase issues by analyzing and correcting before mixing
  - avg: Standard averaging of both channels (L+R)/2
  - left: Uses only the left channel (avoids phase cancellation but loses stereo information)
  - right: Uses only the right channel (same benefit as left-only)

## Additional Features
- Automatic peak normalization to prevent clipping
- Detailed logging of potential phase issues
- Summary report to identify problematic files
- Progress bar for batch processing
