import pandas as pd
import numpy as np
import functools
import time



def diff(mat, rm):
    """
    The function substracts rapidly string from the matrix and returns all the appropriate rows indices.
    :param mat: the initial matrix to substract from (type: np.array)
    :param rm: the row to be substracted (type: np.array)
    :returns: sorted in descending order by number of correct pairs list of rows indices
    """
    # the substraction
    f = mat - rm 
    # filtering values, we actually need just to know, that distance is between 150 and 300, so we use bool type
    Q = np.where((f < 301) & (f > 150) & (mat != 0), True, False) 
    # getting indices of non-zero rows
    inds = np.any(Q, axis=1)
    ins = np.arange(len(Q))[inds]
    # returning the indices with non-zero values filtering in descending order by non-zero values number
    return ins[np.argsort(np.count_nonzero(Q[inds], axis=1))[::-1]]


# Опыт показывает, что паралеллить лучше эту функцию. Опять же, можно использовать разные стратегии 
# распаралелливания, на твоё усмотрение. Это решение предполагает, что мы сначала для каждой строки 
# вернём все строки, образующие пары, отсортированные в порядке убывания, а потом уже в этих списках отдельно, 
# следующим этапом, проверять на гомо.гетеродимеры. Кроме того, можно сначала проверять а гомодимеры и не вычитать
# для них матрицы, это может заметно понизить размер матрицы.
def iterdiff(fmr, fmf, r2c, chunk, start): 
    """
    The function iterates function diff.
    :param fmr (FullMatrixReverse): the initial matrix to substract from consisting from forward 
                                    primers coordinates (type: np.array)
    :param fmf (FullMatrixForward): the reverse matrix for substracting its rows from fmr (type: np.array)
    :param r2c: dictionary where keys - indices of fmf rows, keys - lists of indices of 
                non-zero columns in the row (type: dictionary of lists)
    :param chunk: number of fmf rows to substract from fmr (type: int)
    :param start: number of the first fmf row to start from (type: int)
    :returns: list of tuples, where the first position - index of forward row (int), 
              on the second - list of indices of pair-forming fmr rows, sorted by number of pairs in descending
              order (type: list of tuples)
              example: [(forward index_1, [pair-forming indices_1]),
                        (forward index_2, [pair-forming indices_2]),
                        ...
                        (forward index_n, [pair-forming indices_n])
                        ]
    """
    inds = list() # empty list for results
    for i in range(start, start + chunk): # going through chunk 
        inds.append((i, diff(fmr[:, r2c[i]], nar[i][r2c[i]]))) # adding data to results
    return inds


if __name__ == "__main__":
    print('Reading the matrix...')
    df = pd.read_csv('aln_matrix.csv', index_col='Unnamed: 0') # opening matrix file
    nar = np.array(df, dtype='int32') # making np.array

    print('Starting preparing calculations...')

    r2c = dict.fromkeys(range(nar.shape[0]), None)
    for i in range(nar.shape[0]):
        r2c[i] = np.nonzero(nar[i])[0]
    num_rows = 5000
    f = functools.partial(iterdiff, nar, nar, r2c, num_rows)
    print('Starting calculations...')
    t = time.time()
    rs = f(0)
    a = (time.time() - t)

    print('Finished for {} rows!'.format(num_rows))
    print('Overall time: {} seconds.'.format(time.time() - t)) # total time
    print('Per-row time: {} seconds.'.format((time.time() - t)/num_rows)) # time per row
    time.sleep(1) # sleping 1 second
