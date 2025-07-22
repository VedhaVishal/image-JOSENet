from tqdm import tqdm
import os
import cv2
import glob # Added for more robust path listing

VIDEO_DATASET_ROOT = "D:/imageand/archive/JOSENet/dataset"
OUTPUT_FRAMES_ROOT = "D:/imageand/archive/JOSENet/datasets/ucf_crime_mini_jpg_frames"

# Target frame size for the model
TARGET_DIM = (224, 224) # Ensure this matches what your JOSENet model expects
TARGET_FPS = 7.5 # Target FPS for sampling from original video
CLIP_LEN = 16 # Number of frames per segment JOSENet expects

# --- Helper function to load and process video ---
def load_video_and_extract_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    fps_in = vidcap.get(cv2.CAP_PROP_FPS)
    
    frames_list = [] # List to store all extracted frames
    
    index_in = -1
    index_out = -1 # Keep track of output frames based on TARGET_FPS

    while True:
        success = vidcap.grab() # Grab frame without decoding for speed
        if not success:
            break
        index_in += 1

        # Sample frames based on TARGET_FPS
        out_due = int(index_in / fps_in * TARGET_FPS)
        if out_due > index_out:
            success, frame = vidcap.retrieve() # Decode the grabbed frame
            if not success:
                break
            index_out += 1

            # Process the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
            frame = cv2.resize(frame, TARGET_DIM) # Resize frame
            frames_list.append(frame)

    vidcap.release()
    return frames_list

# --- Main script logic ---

# Ensure output root directory exists
os.makedirs(OUTPUT_FRAMES_ROOT, exist_ok=True)

# Iterate through 'Train' and 'Test' splits
for split_folder in ['Train', 'Test']: # These are your manual split folders
    split_path = os.path.join(VIDEO_DATASET_ROOT, split_folder)
    output_split_dir = os.path.join(OUTPUT_FRAMES_ROOT, 'train' if split_folder == 'Train' else 'val') # JOSENet uses 'train' and 'val' (validation) typically

    # Iterate through categories within each split (e.g., Train/Abuse, Test/Normal)
    categories = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]

    for category in categories:
        category_video_path = os.path.join(split_path, category)
        
        # Get all video files in the current category folder
        # Use glob for more robust video file discovery
        video_files = glob.glob(os.path.join(category_video_path, "*.mp4")) + \
                      glob.glob(os.path.join(category_video_path, "*.avi")) + \
                      glob.glob(os.path.join(category_video_path, "*.mov")) # Add other video extensions if needed

        print(f"Processing {split_folder}/{category} with {len(video_files)} videos...")

        # Process each video in the category
        for video_path in tqdm(sorted(video_files)):
            video_name = os.path.splitext(os.path.basename(video_path))[0] # Get filename without extension

            # Define save directory for this video's frames
            save_dir_for_video = os.path.join(output_split_dir, category, video_name) # Group by category and then video name
            
            if not os.path.exists(save_dir_for_video):
                os.makedirs(save_dir_for_video)
            
            # Load and extract all frames
            extracted_frames = load_video_and_extract_frames(video_path)
            
            # Save extracted frames
            for idx, frame in enumerate(extracted_frames):
                frame_filename = os.path.join(save_dir_for_video, f"{video_name}_{idx:05d}.jpg") # 5-digit padding for frame index
                cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # Convert back to BGR for OpenCV saving

        print(f"Finished processing {split_folder}/{category}.")

print("\nAll videos processed and frames saved!")
print(f"Frames are saved to: {OUTPUT_FRAMES_ROOT}")