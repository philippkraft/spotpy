# -*- coding: utf-8 -*-
'''
Copyright (c) 2017 by Benjamin Manns
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Philipp Kraft

This code shows you, how to use the hydroligcal hydrology. They can also be implemented in the def objective function.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
# from spotpy.examples.spot_setup_hymod import spot_setup
from spotpy.hydrology.signatures import SignatureMethod
import numpy as np


def print_all(data):
    for s in SignatureMethod.find_all():
        print(s)
        print('-' * len(str(s)))
        for var, val in s(data):
            print(f'{var:>10} = {val:0.5g}')
        print()


if __name__ == '__main__':

    fulda = np.recfromcsv('cmf_data/fulda_climate.csv')
    print_all(fulda.q)



