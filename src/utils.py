# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 16:20:19 2017

@author: ylockerman
"""

import bottleneck as bn
from distutils.version import StrictVersion
#Since bottleneck renamed partsort and argpartsort, create a wrapper so we can
#use both versions

if StrictVersion(bn.__version__) >= StrictVersion('1.2.0'):
    def partsort(a, n):
        return bn.partition(a, kth=n-1)
    def argpartsort(a, n):
        return bn.argpartition(a, kth=n-1)
else:
    partsort = bn.partsort
    argpartsort = bn.argpartsort