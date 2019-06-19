import argparse
import pandas as pd
import numpy as np
import math


def normalizeMinMax(min, max, value):
    return (value-min)/(max-min)


def normalizeZScore(mean, stdDeviation, value):
    return (value-mean)/stdDeviation


def normalization(fileName, normalizationType, attribute):
    '''
    Input Parameters:
        fileName: The comma seperated file that must be considered for the normalization
        attribute: The attribute for which you are performing the normalization
        normalizationType: The type of normalization you are performing
    Output:
        For each line in the input file, print the original "attribute" value and "normalized" value seperated by <TAB>
    '''
    # Write code given the Input / Output Paramters.
    ds = pd.read_csv("./" + fileName)
    # Gather stats for normalization
    dsColumn = ds[attribute]
    minimum = float(dsColumn.min())
    maximum = float(dsColumn.max())
    mean = float(dsColumn.mean())
    stdDeviation = float(dsColumn.std(ddof=0))
    # Generate normalization output
    if (normalizationType == 'min_max'): # min-max
        for index, row in ds.iterrows():
            normalizedVal = normalizeMinMax(minimum, maximum, row[attribute])
            print(row[attribute], normalizedVal)
    else: # z-score
        for index, row in ds.iterrows():
            normalizedVal = normalizeZScore(mean, stdDeviation, row[attribute])
            print(row[attribute], normalizedVal)


def correlation(attribute1, fileName1, attribute2, fileName2, start=0, end=None):
    '''
    Input Parameters:
        attribute1: The attribute you want to consider from file1
        attribute2: The attribute you want to consider from file2
        fileName1: The comma seperated file1
        fileName2: The comma seperated file2
        start: The number of beginning rows from each dataset to skip (starting at index 0)
        end:   The last row from each dataset to correlate (starting at index 0)

    Output:
        Return the correlation coefficient
    '''
    # Read the datasets and slice them with start and end.
    ds1 = pd.read_csv("./" + fileName1)
    ds2 = pd.read_csv("./" + fileName2)
    if end is None:
        ds1 = ds1.iloc[start:]
        ds2 = ds2.iloc[start:]
    else:
        ds1 = ds1.iloc[start:end]
        ds2 = ds2.iloc[start:end]

    # Gather stats for correlation
    dsColumn1 = ds1[attribute1]
    dsColumn2 = ds2[attribute2]
    n = int(dsColumn1.count()) #TODO: assert that both columns have same count?
    meanA = float(dsColumn1.mean())
    meanB = float(dsColumn2.mean())
    stdA = float(dsColumn1.std(ddof=0))
    stdB = float(dsColumn2.std(ddof=0))
    # Generate correlation output
    dsJoined = pd.concat([dsColumn1, dsColumn2], axis=1)
    dsJoined.columns = ['a', 'b']  # avoids ambiguity when both attribute names are the same
    numerator = 0.0  # stores summation of (a_i - meanA)(b_i - meanB)
    denominator = n * stdA * stdB

    for index, row in dsJoined.iterrows():
        a = row['a']
        b = row['b']
        if (not math.isnan(a) and not math.isnan(b)):  # ignore any row with a NaN
            numerator = numerator + (a - meanA)*(b - meanB)

    correlationCoefficient = numerator/denominator
    return correlationCoefficient


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Mining HW2')
    parser.add_argument('-f1', type=str,
                            help="Location of filename1. Use only f1 when working with only one file.",
                            required=True)
    parser.add_argument("-f2", type=str,
                            help="Location of filename2. To be used only when there are two files to be compared.",
                            required=False)
    parser.add_argument("-n", type=str,
                            help="Type of Normalization. Select either min_max or z_score",
                            choices=['min_max','z_score'],
                            required=False)
    parser.add_argument("-a1", type=str,
                            help="Type of Attribute for filename1. Select either open or high or low or close or volume",
                            choices=['open','high','low','close','volume'],
                            required=False)
    parser.add_argument("-a2", type=str,
                            help="Type of Attribute for filename2. Select either open or high or low or close or volume",
                            choices=['open','high','low','close','volume'],
                            required=False)



    args = parser.parse_args()

    if ( args.n and args.a1 ):
        normalization( args.f1 , args.n , args.a1 )
    elif ( args.f2 and args.a1 and args.a2):
        correlation ( args.a1 , args.f1 , args.a2 , args.f2 )
    else:
        print ( "Kindly provide input of the following form:\nDMPythonHW2.py -f1 <filename1> -a1 <attribute> -n <normalizationType> \nDMPythonHW2.py -f1 <filename1> -a1 <attribute> -f2 <filename2> -a2 <attribute>" )
