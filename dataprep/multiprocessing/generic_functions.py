# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 09:54:03 2022

@author: Mike
"""

"""
List of generic functions that are SLOW and need to be multiprocessed

    - Generating the train data
        - Calculating target values
        - Creating new arrays; y, X
    
    
    The two sinners:
        
    - Padding the train data
    - Creating the inference tables


Can these be made in a simpler way?

"""


"""
List comprehension
"""

# List comprehensions are fast in some cases
def faster():
    return [elem for elem in some_iterable]


numbers = list(range(0,2000))

def transform(a):
    b = a*200
    return b

res = [transform(elem) for elem in numbers]
res[0:10]


"""
Vectorization
"""


"""
Joins are fast!
"""


"""
Multiprocessing
"""