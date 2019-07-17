'''
This is the main file for preprocessing Emilya dataset.
: retrieve all datapoints and labels includes their actor label into hdf5 file
'''

__author__ = 'qiwang'
import h5py
import numpy as np

import sys
import os
root_path = os.getcwd()[:-37]  # Gets the path above the root directory
sys.path.append(root_path)

# from Seq_AAE_V1.bvh_tools.bvhplay.skeleton import *
from Seq_AAE_V1.bvhplay.skeleton import *  # Edited path


from os import listdir
from os.path import isfile,join
from os import getenv,environ
# home_path = environ['HOME']
# project_path =getenv('Seq_AAE_V1')
project_path = root_path+'Seq_AAE_V1/'
home_path = root_path
save_path = project_path+'datasets/EmilyaDataset/'
data_path = home_path+'Thesis/Dataset/EmilyaDatabase/All_BvH_Files/'
data_savepath = project_path+'datasets/EmilyaDataset/'
print data_path
print '========'

Activity_name = ['BS', 'Lf', 'SW', 'Th', 'KD', 'MB', 'SD', 'WH']
Emotion_name = ['Anger','Anxiety','Joy','Neutral','Panic Fear','Pride','Sadness','Shame']
Actor_name = ['Actor1', 'Actor2', 'Actor3', 'Actor4', 'Actor5', 'Actor6', 'Actor7', 'Actor8', 'Actor9', 'Actor10', 'Actor11',
            'Actor12']

print len(Activity_name)
#trainingset =[]
keyframes=[]
i_array=[]
j_array = []
k_array = []
consonants = set("bcdfghjklmnpqrstvwxyz")
for i in xrange(len(Activity_name)):
    for j in xrange(len(Emotion_name)):
        for k in xrange(len(Actor_name)):
            curpath = join(data_path,Activity_name[i],Emotion_name[j],Actor_name[k])
            #curpath = data_path+Activity_name[i]+'/'+Emotion_name[j]+'/'+Actor_name[k]+'/'
            print curpath
            filelist = listdir(curpath)
            #print filelist
            for ii,filename in enumerate(filelist):
                if filename[-3:] != 'bvh':
                    del filelist[ii]
                    continue
                if filename[-7:]=='zyx.bvh':
                    print 'delete zyx file:%s'%filename
                    del filelist[ii]
                    continue
                fullname = join(curpath,filename)
                bvh = readbvh(fullname)
                try:
                    bvh.read()
                except:
                    print('Warning: Error occurred file: {}.'.format(fullname))
                keyframes.append(np.asarray(bvh.keyframes,dtype=np.float32))
                i_array.append(i)#activity label
                j_array.append(j)#emotion label
                k_array.append(k)#actor label

#if (i+1)%2 == 0:
    #dump trainingset to a pickle file
    #trainingset = np.asarray(trainingset,dtype=np.float32)
assert len(keyframes)==len(i_array)
print 'length of keyframes is %d'%len(keyframes)
keyframes = keyframes
print keyframes[0].shape
i_array = np.asarray(i_array,dtype=np.float32)
j_array = np.asarray(j_array,dtype=np.float32)
k_array = np.asarray(k_array, dtype=np.float32)
dataset = (keyframes,i_array,j_array,k_array)
filename = save_path+'Emilya_Dataset.npz'
np.savez(filename,dataset)
