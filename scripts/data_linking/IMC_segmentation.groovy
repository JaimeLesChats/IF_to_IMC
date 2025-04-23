

// Load image and detect cells, then export measurements

setImageType('FLUORESCENCE') // or BRIGHTFIELD depending on your data
selectAnnotations()

if (getAnnotationObjects().isEmpty()) {
    createSelectAllObject(true)
}

// Cell detection (adjust parameters)
runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', '''
{
  "requestedPixelSizeMicrons": 0.5,
  "backgroundRadiusMicrons": 8.0,
  "medianRadiusMicrons": 0.0,
  "sigmaMicrons": 2,
  "minAreaMicrons": 5.0,
  "maxAreaMicrons": 400.0,
  "threshold": 2.0,
  "cellExpansionMicrons":0.001,
  "watershedPostProcess": true,
  "includeNuclei": false,
  "smoothBoundaries": true,
  "makeMeasurements": true
}
''')

// Export detections as CSV
def path = '/home/matthieu.bernard/Documents/IF_to_IMC/data/true_data/IMC_morphology'
mkdirs(path)
saveDetectionMeasurements(buildFilePath(path, getProjectEntry().getImageName() + '_measurements.csv'))