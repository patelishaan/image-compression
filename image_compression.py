import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from skimage import io
import seaborn as sns
from PIL import Image

window = 'output'
myimg = io.imread('turing.jpg')
gray_img = cv.cvtColor(myimg, cv.COLOR_BGR2GRAY)
img_mat = np.array(list(gray_img),float)
print(img_mat)

plt.imshow(img_mat, cmap = plt.cm.gray)
plt.axis('off')
print(img_mat.shape)

plt.savefig('original.png',bbox_inches = 'tight', pad_inches = 0)
plt.show()


img_mat_scaled = (img_mat-img_mat.mean())/img_mat.std()


#SVD
u, s, v = np.linalg.svd(img_mat_scaled)
#variance
#var=np.round(s**2/np.sum(s**2),decimals=3)
#sns.barplot(x=list(range(1,11)),
            #y=var[0:10],color='dodgerblue')
#plt.xlabel('singular vector',fontsize = 16)
#plt.ylabel('variance',fontsize = 16)
#plt.tight_layout()
#plt.show()


#apply
num = 45
reconstruction = np.array(u[:,:num]).dot(np.diag(s[:num]).dot(np.array(v[:num,:])))
print(reconstruction.shape)
plt.imshow(reconstruction, cmap = plt.cm.gray)
plt.axis('off')
plt.savefig('compressed.png', bbox_inches = 'tight', pad_inches = 0)
plt.show()
#cv.imshow(window,reconstruction)
#cv.waitKey(0)
#cv.destroyAllWindows()