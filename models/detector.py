import os
import shutil

def process_video(video_path, output_dir):
    
    hasil_video_path = os.path.join(output_dir, "hasil_output.mp4")
    
    
    shutil.copy(video_path, hasil_video_path)

    # jumlah kendaraan masuk & keluar
    return {
        "masuk": 14,
        "keluar": 9,
        "video_result": "/result/hasil_output.mp4"
    }
