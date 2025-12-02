import os
from PIL import Image, ImageFile
from multiprocessing import Pool
from tqdm import tqdm

IMG_ROOT = "images"
ImageFile.LOAD_TRUNCATED_IMAGES = False
NUM_PROCS = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))

def check_image(file):
    path = os.path.join(IMG_ROOT, file)
    try:
        img = Image.open(path)
        img.verify()
        return None  # OK
    except:
        return path  # bad file path

files = os.listdir(IMG_ROOT)

with Pool(processes=NUM_PROCS) as pool:   # choose ~ #CPU cores
    results = list(tqdm(pool.imap(check_image, files), total=len(files)))

# remove bad files
bad_files = [r for r in results if r is not None]

print(f"Found {len(bad_files)} bad images")

for bf in tqdm(bad_files):
    print("Removing:", bf)
    os.remove(bf)
print("Done removing bad images.")