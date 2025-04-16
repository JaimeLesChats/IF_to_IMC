import tifffile as tiff
import matplotlib.pyplot as plt
from PIL import Image

def read_and_crop(cellbound,i):
    return tiff.imread('./data/raw/Vizgen/test/mosaic_{}_z{}.tif'.format(cellbound,i))[5800:10300,6100:10400]

marker = 'GFP'
imgs = [read_and_crop(marker,i) for i in range(6)]


# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(9, 9))

# Flatten the axes array for easy indexing
axes = axes.ravel()

# Show images
for i, img in enumerate(imgs):
    axes[i].imshow(img)
    axes[i].set_title(f"Image {i+1}")
    axes[i].axis("off")

plt.show()

