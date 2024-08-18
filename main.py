import concurrent.futures
import time
import cv2
import os




# Create a FLANN matcher object
index_params = dict(algorithm=6,  # LSH
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

cache = {}
sift = cv2.ORB_create()

def match_image(desB, imageA_data):
    """Match features of Image A against Image B."""
    filename, (_, desA) = imageA_data
    if desA is None or desB is None:
        return None

    matches = flann.knnMatch(desA, desB, k=2)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:  # Ensure there are two matches to unpack
            m, n = match_pair
            if m.distance < 0.35 * n.distance:
                good_matches.append(m)

    if len(good_matches) > 0:
        return filename, good_matches
    return None

def find_image_in_image(imageB_path):
    print("Checking image..")
    start = time.time()
    imgB = cv2.imread(imageB_path, 0)  # Load as grayscale

    if imgB is None:
        print(f"Could not load image: {imageB_path}")
        return None

    _, desB = sift.detectAndCompute(imgB, None)

    best_match = [0, None]

    print("Finding best match...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(match_image, desB, data) for data in cache.items()]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                filename, good_matches = result
                if len(good_matches) > best_match[0]:
                    best_match = [len(good_matches), filename]

    if best_match[1] is not None:
        return (best_match[1].split(".png")[0]).split("_flipped")[0], round(time.time() - start, 2)
    else:
        print("No match found.")
        return None, round(time.time() - start, 2)

def process_image(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        return None
    kp, des = sift.detectAndCompute(img, None)
    return (kp, des)

print("Loading Images...")
start = time.time()
images = [os.path.join("dataset", filename) for filename in os.listdir("dataset")]
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(process_image, images))
    for filename, result in zip(os.listdir("dataset"), results):
        if result:
            kp, des = result
            cache[filename] = [kp, des]
print(f"Successfully loaded all images...\nTime Taken: {round(time.time()-start, 2)} sec")
print("--------------------------------")

# Example usage
imageB_path = 'predict/pikachu.png'
matching_image, taken_time = find_image_in_image(imageB_path)

if matching_image:
    print(f"Input: {imageB_path}\nMatch: {matching_image}\nTime Taken: {taken_time} sec\n\n")
else:
    print(f"No match found.\nTime Taken: {taken_time} sec\n\n")

while True:
    imageB = input("- Which image to check now?\n ")
    matching_image, time_taken = find_image_in_image(f"predict/{imageB}")

    if matching_image:
        print(f"Input: {imageB}\nMatch: {matching_image}\nTime Taken: {time_taken} sec\n\n")
    else:
        print(f"No match found.\nTime Taken: {time_taken} sec\n\n")
