import os
import random,string
import sys

"""
Change file names to random
"""
def randomword(length):
   return ''.join(random.choice(string.lowercase) for i in range(length))

image_dir = sys.argv[1]

# recursively explore directory
for root,dir,files in os.walk(image_dir):
    for file in files:
        if 'jpg' in file or 'JPG' in file:
            os.rename(os.path.join(root,file), os.path.join(root,randomword(10)+'.jpg'))
