''' This script downloads the images of the given class '''
import json
import urllib 
from tqdm import *
import ipdb
import argparse 
import urllib.request
import os, sys
from timeout_decorator import timeout
from random import shuffle

url_retrieve_timeout = 30
  # Adjust the value as needed

def retrieve_url_with_timeout(url, filename):
    urllib.request.urlretrieve(url, filename)


parser = argparse.ArgumentParser(description="Download images to given folder")
parser.add_argument("-c", type = str, help = "The class whose images will be downloaded")
parser.add_argument("-n", type = str, default = "all", help = "The number of images to be downloaded (with auto upper limit)")
parser.add_argument("-o", type = str, help = "output folder for downloaded images")
args = parser.parse_args()

# load url file 
with open("fall11_urls.txt","r", encoding='latin-1') as f:
    urls = f.readlines() 
    urls = [str(url.strip()) for url in urls]
    

# load class to wnid 
with open("imagenet_class_index.json","r") as f:
    class_index = json.loads(f.read())
class_to_wnid = {}
for i in range(1000):
    wnid, class_name = class_index[str(i)]
    class_to_wnid[class_name.lower()] = wnid

# Split urls and image ids
class_name = args.c 
class_wnid = str(class_to_wnid[class_name])
urls_wnid  = [url.split("\t")[-1] for url in urls if class_wnid in url]
imageids   = [url.split("\t")[0] for url in urls if class_wnid in url]
assert len(urls_wnid) == len(imageids), "urls_wnid and imageids do not have the same length"
print ("[LOG]: Classname: {}".format(class_name))
print ("[LOG]: WordNet ID: {}".format(class_wnid))


# Number of images to download 
num_images = 0
if args.n == "all":
    num_images = len(imageids)
elif int(args.n) > len(imageids):
    print ("[WARN]: User argument 'number of images' exceeds max images in dataset, Auto capping to MAX")
    num_images = len(imageids)
else:
    num_images = int(args.n)
print ("[LOG]: Downloading {} images out of a total of {}".format(num_images, len(imageids)))

# ipdb.set_trace()
# Shuffle  
shuff_idxs = range(len(imageids))
#shuffle(shuff_idxs)
urls_wnid = [urls_wnid[i] for i in shuff_idxs][:num_images]
imageids =  [imageids[i] for i in shuff_idxs][:num_images]

# Download folder 
output_folder = args.o 
assert os.path.isdir(output_folder), "Invalid output folder path provided by user"
print ("[LOG]: output folder: {}".format(output_folder))

# Downloading images from url
item = 1
for url, imid in tqdm(zip(urls_wnid, imageids)):
    if(item > 500):
        fname = imid 
        if url.strip().split(".")[-1].lower() in ['jpg', 'gif', 'jpeg','png', 'bmp', 'thb', 'jpe', 'tif', 'pjpeg']:
            fname = fname + "." + url.strip().split(".")[-1] # keep the file extension in fname 
        else: 
            fname = fname + ".jpg" # default use .jpg file extension
        try:
            retrieve_url_with_timeout(url, os.path.join(output_folder, fname))
        except TimeoutError:
            print("URL retrieval timed out for url:", url, " | skipping")
        except Exception as e:
            print("An error occurred while retrieving url:", url, "|", e, " | skipping")
    item += 1

