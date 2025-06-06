import qupath.lib.objects.PathAnnotationObject
import qupath.lib.roi.RectangleROI
import org.yaml.snakeyaml.Yaml

Yaml parser = new Yaml()
/*List example = parser.load(("../../config.yaml" as File).text)*/

// Annotations

double pixel_size = 0.226

def roi_list = [[16686.59,5875,2589,1612]]

for (r in roi_list) {
    def roi = new RectangleROI((r[0]-r[2])/pixel_size, r[1]/pixel_size, r[2]/pixel_size, r[3]/pixel_size)

    def annotation = PathObjects.createAnnotationObject(roi)
    addObject(annotation)
    selectObjects(annotation)
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