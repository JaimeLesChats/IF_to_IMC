// Ensure an image is open
def imageData = getCurrentImageData()
if (imageData == null) {
    print "No image open!"
    return
}

// Load the InstanSeg extension
def instanseg = qupath.ext.instanseg.InstanSegRunner.getInstance()

// Set model and parameters
instanseg.setModelName("brightfield_nuclei")  // Or "fluorescence_nuclei"
instanseg.setTileSize(1024)                   // Optional: change tile size
instanseg.setOverlap(64)                      // Optional: tile overlap
instanseg.setUseGPU(true)                     // Or false if CPU
instanseg.setNormalize(true)                  // Whether to normalize intensity

// Run segmentation
instanseg.run(imageData)

// Export detections as CSV
def path = '/home/matthieu.bernard/Documents/IF_to_IMC/data/true_data/morphology/IMC_morphology'
mkdirs(path)
saveDetectionMeasurements(buildFilePath(path, getProjectEntry().getImageName() + '_measurements.csv'))