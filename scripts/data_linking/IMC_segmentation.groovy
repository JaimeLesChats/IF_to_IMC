

// Load image and detect cells, then export measurements

setImageType('FLUORESCENCE') // or BRIGHTFIELD depending on your data
selectAnnotations()

if (getAnnotationObjects().isEmpty()) {
    createSelectAllObject(true)
}

// Cell detection (adjust parameters)
runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', '''
{
  'detectionImage': 'DNA1', 
  "backgroundRadiusMicrons": 8.0,
  "medianRadiusMicrons": 0,
  "sigmaMicrons": 1.1,
  "minAreaMicrons": 4.0,
  "maxAreaMicrons": 200.0,
  "threshold": 20.0,
  "cellExpansion": 0.000,
  "watershedPostProcess": true,
  "includeNuclei": true,
  "smoothBoundaries": true,
  "makeMeasurements": true
}
''')

// Export detections as CSV
def path = '/home/matthieu.bernard/Documents/IF_to_IMC/data/true_data/morphology/IMC_morphology'
mkdirs(path)
saveDetectionMeasurements(buildFilePath(path, getProjectEntry().getImageName() + '_measurements.csv'))