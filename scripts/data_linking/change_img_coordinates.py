import pandas as pd

qwidth = 24081.59
qheight = 18640.07
qorigin = [12142,4880]

def n(coords):
    return [qwidth-(coords[0] - qorigin[0]),(coords[1]-qorigin[1])]

roi_data = pd.read_excel('22-IMC-H-27_roi.xlsx')

#df = pd.read_csv('15T011146-16 HE - 2022-06-09 13.57.csv')

roi1 = [19537,10755]
roi2 = [24079,11590]
roi3 = [26589,11433]


print('roi1 : {}\nroi2 : {}\nroi3 : {}\n'.format(n(roi1),n(roi2),n(roi3)))