import random
import subprocess
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm


def convert_to_wav(src_file_path):
    dst_file_path = dst_dir / src_file_path.relative_to(src_dir).with_suffix(".wav")
    dst_file_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_file_path.exists():
        return True

    try:
        subprocess.check_call(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src_file_path),
                "-c:a",
                "pcm_s16le",
                "-threads",
                "0",
                "-ar",
                "24000",
                str(dst_file_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # remove the output file
        dst_file_path.unlink()
        return True
    except subprocess.CalledProcessError:
        return False


src_dir = Path("dataset/tts/WenetSpeech/audio/")
dst_dir = Path("dataset/tts/WenetSpeech/audio_wav/")

opus_files = list(src_dir.rglob("*.opus"))
random.shuffle(opus_files)
print(f"Found {len(opus_files)} opus files")

success_counter = 0
fail_counter = 0

with Pool(processes=cpu_count() * 2, maxtasksperchild=100) as pool:
    with tqdm(
        pool.imap_unordered(convert_to_wav, opus_files), total=len(opus_files)
    ) as pbar:
        for success in pbar:
            if success:
                success_counter += 1
            else:
                fail_counter += 1

        pbar.set_description(f"Success: {success_counter}, Fail: {fail_counter}")

print(f"Successfully converted: {success_counter}")
print(f"Failed conversions: {fail_counter}")
