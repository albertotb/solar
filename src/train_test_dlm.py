import pandas as pd
import numpy as np
import feather
import datetime
from pydlm_lite import dlm, trend

df = (feather.read_dataframe('/home/SHARED/SOLAR/data/oahu_min.feather')
             .set_index('Datetime'))
df[df < 0] = 0

## Try DLM in one sensor

GH_DH4 = df.GH_DH4
train = GH_DH4[:-50]
test = GH_DH4[-50:]
##
dlm_ghdh4 = dlm(train)
dlm_ghdh4 = dlm_ghdh4 + trend(degree=2, discount=0.90, name='ltrend', w=10.0) ## Just a linear trend

dlm_ghdh4.fitForwardFilter()

preds = np.zeros(len(test))
for i in range(len(test)):
    preds[i] = dlm_ghdh4.predictN(1)[0][0]
    ##
    dlm_ghdh4.append([test[i]])
    dlm_ghdh4.fitForwardFilter()

np.savetxt('predictions.out', (test,preds))   # x,y,z equal sized 1D arrays
mae =np.mean( np.abs(test.values - preds) )
print(mae)
