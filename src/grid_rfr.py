#!/bin/env python

import sys
import os
import grp
import subprocess
from itertools import product

HOME='/LUSTRE/users/atorres/'

if len(sys.argv) < 1 or len(sys.argv) > 2:
    print("usage: {0} OUTDIR".format(sys.argv[0]))
    sys.exit(1)

if len(sys.argv) == 2:
    outdir = sys.argv[1]
else:
    outdir = '.'

grid = product(['../../data/oahu_min.feather'],
               [11],
               range(19),
               [1234])

for idx, params in enumerate(grid):
    # run job in GSE and return jobid
    name = 'rfr-{0:03d}'.format(idx)
    fout = '{0}/{1}.out'.format(outdir, name)
    args = ['qsub', '-q', 'all.q', '-cwd', '-j', 'y', '-o', fout, '-V', '-l', 'h_vmem=4G', '-N', name,
            HOME+'pywrap.sh', 'train_rfr.py'] + [ str(param) for param in params ]
    print(' '.join(args))
    output = subprocess.Popen(args, stdout=subprocess.PIPE).stdout.read()
    print(output, params)
    break
