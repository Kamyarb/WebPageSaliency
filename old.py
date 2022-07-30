
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import random
import time
import warnings
from os import listdir
from os.path import isfile, join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import skimage
from matplotlib import pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter
from skimage import color, data, io
from skimage.feature import canny
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential

warnings.filterwarnings("ignore")
import seaborn as sns; sns.set()
pictureNames = [f for f in listdir('./dataset/stimuli/') \
    if isfile(join('./dataset/stimuli/', f)) if not f.startswith('Thumb')]
rawData_eyelink = [f for f in listdir('./dataset/raw/') \
    if isfile(join('./dataset/raw/', f)) if f.endswith('.asc')]


        
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def saccadic(elFilename):
    f = open(elFilename,'r')
    fileTxt0 = f.read().splitlines(True) 
    nLines = len(fileTxt0)
    lineType = np.array(['OTHER']*nLines,dtype='object')
    for iLine in range(nLines):
        if len(fileTxt0[iLine])<3:
            lineType[iLine] = 'EMPTY'
        elif fileTxt0[iLine].startswith('*') or fileTxt0[iLine].startswith('>>>>>'):
            lineType[iLine] = 'COMMENT'
        elif fileTxt0[iLine].split()[0][0].isdigit() or fileTxt0[iLine].split()[0].startswith('-'):
            lineType[iLine] = 'SAMPLE'
        else:
            lineType[iLine] = fileTxt0[iLine].split()[0]
        if '!CAL' in fileTxt0[iLine]: 
            iStartRec = iLine+1
    nLines = len(fileTxt0)
    lineType = np.array(['OTHER']*nLines,dtype='object')
    iStartRec = None
    t = time.time()
    for iLine in range(nLines):
        if len(fileTxt0[iLine])<3:
            lineType[iLine] = 'EMPTY'
        elif fileTxt0[iLine].startswith('*') or fileTxt0[iLine].startswith('>>>>>'):
            lineType[iLine] = 'COMMENT'
        elif fileTxt0[iLine].split()[0][0].isdigit() or fileTxt0[iLine].split()[0].startswith('-'):
            lineType[iLine] = 'SAMPLE'
        else:
            lineType[iLine] = fileTxt0[iLine].split()[0]
        if '!CAL' in fileTxt0[iLine]: 
            iStartRec = iLine+1


    iNotEfix = np.nonzero(lineType!='EFIX')[0]
    dfFix = pd.read_csv(elFilename,skiprows=iNotEfix,header=None,delim_whitespace=True,
                        usecols=range(1,8))
    dfFix.columns = ['eye','tStart','tEnd','duration','xAvg','yAvg','pupilAvg']
    nFix = dfFix.shape[0]
    mapper = {}
    mapper_name = {}
    for iLine in range(nLines):
        if 'START' in fileTxt0[iLine]:
            x = fileTxt0[iLine]
            try:
                numberofevent = int(x.split(' ')[-2])
                timestamp = int(x.split(' ')[0].split('\t')[1])
                
            except: continue
            mapper[x.split(' ')[-2]] = \
                {'timestamp' : x.split(' ')[0].split('\t')[1],
                 'filename' : x.split(' ')[-1].replace('\n','')}
    df_mapper= pd.DataFrame(mapper).T
    df_mapper['timestamp'] = df_mapper['timestamp'].astype(int)
    
    def findEventNumber(x):
        try:
            return df_mapper['filename'][(x > df_mapper.timestamp)].iloc[-1]
        except: 
            return df_mapper['filename'].iloc[0]
    dfFix['stimuliName'] = dfFix['tStart'].apply(findEventNumber)
    # return dfFix
    iNotEsacc = np.nonzero(lineType!='ESACC')[0]
    dfSacc = pd.read_csv(elFilename,skiprows=iNotEsacc,header=None,delim_whitespace=True,usecols=range(1,11))
    dfSacc.columns = ['eye','tStart','tEnd','duration','xStart','yStart','xEnd','yEnd','ampDeg','vPeak']
    # dfSacc = dfSacc.iloc[:10]
    dfSacc['stimuliName'] = dfSacc['tStart'].apply(findEventNumber)
    dfSacc['order']=0
    for event in dfSacc.stimuliName.unique().tolist():
        dfSacc['order'][dfSacc.stimuliName==event] =\
            200* (0.8)** np.arange(1, dfSacc[dfSacc.stimuliName==event].shape[0]+1)
    return dfSacc, dfFix
def gaussuian_filter(kernel_size, sigma=3, muu=0 ,value=1):
 
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
 
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)
    # lower normal part of gaussian
    # normal = 1/np.sqrt(2 * np.pi * sigma**2)
 
    # Calculating Gaussian filter
    gauss = np.exp(-((5*dst-muu)**2 / (2.0 * sigma**2))) * value
    return gauss




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# i = 0 
# for data in rawData_eyelink:
#     elFilename = './dataset/raw/'+ data
#     if i==0:
#         allSacadic, allFix = saccadic(elFilename)
#         allSacadic['name'] = data
#         allFix['name'] = data
        
#         i += 1
#     else:
#         print(data)
#         sacadic, fix  = saccadic(elFilename)
#         fix['name'] = data
#         sacadic['name'] = data
#         allSacadic = pd.concat([allSacadic, sacadic])
#         allFix = pd.concat([allFix, fix])
        
    
allSacadic= pd.read_csv('sacadic.csv')
allSacadic = allSacadic.drop('Unnamed: 0',axis=1)
allFix = pd.read_csv('fixations.csv')
allFix = allFix.drop('Unnamed: 0',axis=1)
# %%
# allFixations.groupby(['stimuliName'])['xAvg', 'yAvg'].count()





# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# for name in pictureNames:

name  = pictureNames[67]
# name = 'ieee.png'

pts = allSacadic[allSacadic['stimuliName']==f'{name}'][['xStart','yStart','order']].astype(int).values
pts_fix= allFix[allFix['stimuliName']==f'{name}'][['xAvg','yAvg','duration']].round().astype(int).values

#%%%

img = plt.imread(f'./dataset/stimuli/{name}')
mesh = np.zeros((img.shape[0], img.shape[1]))
print(name)
plt.figure(figsize=(15,15))
# grayscale = color.rgb2gray(img) 
# img_canny= canny(grayscale)

plt.imshow(img)
# plt.imshow(canny(img[:,:,2]), cmap='gray')
# plt.imshow(heatmap)
o=plt.scatter(pts[:, 0], pts[:, 1], marker="o", 
            color="red", s=1*pts[:,2], cmap='hot', alpha=0.5)
o2=plt.scatter(pts_fix[:, 0], pts_fix[:, 1], marker=".", 
            color="green", s=pts_fix[:,2], cmap='hot', alpha=0.2)
plt.legend((o, o2),('Saccadic','Fixation'),loc='lower left')
plt.axis('off')
plt.show()
# time.sleep(6)


#%%%%%%%%%%%
name  = pictureNames[67]
# name = [x for x in pictureNames if x.find('google_news')!=-1][0]

pts = allSacadic[allSacadic['stimuliName']==f'{name}'][['xStart','yStart','order']].astype(int).values
pts_fix= allFix[allFix['stimuliName']==f'{name}'][['xAvg','yAvg','duration']].round().astype(int).values
img = plt.imread(f'./dataset/stimuli/{name}')

# %%
x_dim, y_dim ,_=img.shape
heatmap = np.zeros((x_dim, y_dim), dtype=float)

for vector in pts_fix:
    try:
        x = vector[1]
        y = vector[0]
        order  = vector[2]
        m = int(70*np.tanh(order))
        # m = int(order)
        heatmap[x-m:x+m, y-m:y+m] += gaussuian_filter(2*m, sigma=2,value=order)
    except:pass
print(name)
heatmap /= heatmap.max()
heatmap *= 255
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plt.figure(figsize=(15,15))
plt.axis('off')
plt.imshow(img, alpha=.6)
# plt.imshow(cv2.filter2D(heatmap, -1, np.ones((3,3))) , cmap='jet', alpha=0.4)

plt.imshow(heatmap , cmap='jet', alpha=0.5)


# %%

# %%
def finding_sailentandnonsailent(heatmap):
    result = np.where(heatmap > np.percentile(heatmap,80))

    sailentindices = list(zip(result[0], result[1]))
    result = np.where(heatmap <= np.percentile(heatmap,30))

    nonsailentindices = list(zip(result[0][:int(.3* heatmap.size)], 
                                result[1][:int(.3* heatmap.size)]))
    return sailentindices, nonsailentindices




# %%
finding_sailentandnonsailent(heatmap)
# %%
