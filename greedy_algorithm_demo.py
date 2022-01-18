import pandas as pd
import numpy as np
import functools
import time
import math
import primer3


def sort_by_nz(df):
    '''
    The function sorts rows of pandas DataFrame by descending of number of non-zero values and removes
    all full-zero rows.
    :param df: DataFrame needed to be sorted (type: Pandas DataFrame)
    :returns: df with rows sorted by descending of number of non-zero values and removed
    full-zero rows (type: Pandas DataFrame)
    '''
    df2 = df.copy() # making copy of input DataFrame
    # making additional column with number of nonzero values in rows
    df2['Max_nz'] = df2.apply(lambda x: np.count_nonzero(x), axis=1) 
    # sorting by descending of number of nonzero values (value in 'Max_nz' column)
    df2.sort_values(by=['Max_nz'], ascending=False, inplace=True)
    # full-zero rows removal
    df2 = df2[df2['Max_nz'] > 0].iloc[:, :-1]
    return df2


# эта функция легко может быть заменена на функцию calcTm из пакета primer3,
# однако я по умолчанию использую в своих проектах именно эту, 
# поскольку я заметил огромную разницу между результатами, 
# выдаваемыми primer3.calcTm и ThermoFisher Multiple primer analyzer.  
# Исторически, именно последняя программа использовалась мной для 
# разработки наборов праймеров, поэтому я реализовал эту программу локально
# и использую её по умолчанию. Поэтому я оставляю её применение на усмотрение пользователя.
def Tm(seq, salt=0.05, pri=0.000005):
    '''
    The function calculates oligo Tm according to ThermoFisher Multiple primer analyzer (NN method).
    It needs dH and dS arrays saved in repository.
    :param seq: the oligo sequence in the upper case (type: str)
    :param salt: the salt concentration (type: float)
    :param pri: the oligo concentration (type: float)
    :returns: Tm of the oligo (type: float)
    '''
    seq = seq.lower()
    dHf = -3400
    dSf = -12.4
    dH = np.load('dH.npy')
    dS = np.load('dS.npy')
    cod_dict = dict(zip([0, 1, 2, 3, 6, 7, 10, 12, 13, 17, 18, 19, 21, 22, 23, 24], range(16)))
    for i in range(len(seq)-1):
        k1 = cod_dict[ord(seq[i]) - 97]
        k2 = cod_dict[ord(seq[i+1]) - 97]
        dHf+=dH[k1][k2]
        dSf+=dS[k1][k2]
    T = 16.6 * math.log10(salt) + dHf / (1.9872 * math.log(pri / 1600) + dSf) - 273.15
    return T



def diff2(df1, df2, ind, chem_min=150, chem_max=300): 
    '''
    The function substracts df2.loc[ind] row from each row of df1, if initial value 
    in df1 or in df2 is 0 or their difference shorter than 150 or longer than 300, 
    then returns 0 in this position.
    :param df1: the first DataFrame to substract from (type: Pandas DataFrame)
    :param df2: the second DataFrame to substract its row (type: Pandas DataFrame)
    :param ind: the index of df2 row to substract from d1 (type: str)
    :returns: the sorted and filtered result of described substraction (type: Pandas DataFrame)
    '''
    r = df1 - df2.loc[ind] # actually substraction
    # filtering by length and initial value of df1
    r.where((r < chem_max + 1) & (r > chem_min) & (df1 != 0), other=0, inplace=True)
    # filtering by value of df2.loc[ind]
    r = r * df2.loc[ind].astype('bool')
    # sorting rows by descending of number of nonzero values, dropping all-zeros rows
    r = sort_by_nz(r)
    return r


def ishet(seq1, seq2):
    '''
    The function checks the existance of hetrodimer between seq1 and seq2 oligos with primer3 tool.
    :param seq1: the first nuclic acid sequence (type: str)
    :param seq2: the second nuclic acid sequence (type: str)
    :returns: True if dG of heterodimer less than -2000, else False (type: bool)
    '''
    hod = primer3.calcHeterodimer(seq1, seq2) # calculating structure with primer3
    if not hod.structure_found or hod.dg > -2000:
        return False
    else:
        return True

def no_hdims(pool, seq):
    '''
    The function checks abscence of hetrodimers between the initial pool oligos and the new candidate oligo.
    :param pool: pool of the oligo sequences (type: list of str)
    :param seq: the candidate oligo sequence (type: str)
    :returns: if there is at least one heterodimer, then False, else True (type: bool)
    '''
    # acecking dimers with all the sequences in pool
    for primer in pool: 
        if ishet(primer, seq):
            return False
    
    return True

def index_nhd(pool, index):
    '''
    The function returnes all indices (oligo sequences) which do not form heterodimers with presented in pool.
    :param index: all indices in initial DataFrame  (type: list of str)
    :param pool: pool of the oligo sequences (type: list of str)
    :returns: list of oligos non-dimer-forming with pool oligos (type: list of str)
    '''
    indices = list() # empty list of non-dimer-forming indices
    for ind in index: # choosing only indices without dimers
        if no_hdims(pool, ind):
            indices.append(ind)
    return indices


    


def make_pairs(rdf, fdf, pool):
    '''
    The function gets forward primer DataFrame, reverse primer DataFrame and pool of primers already
    chosen. It returns covered genomes and covering primers.
    :param rdf: forward primer DataFrame (type: Pandas DataFrame)
    :param fdf: reverse primer DataFrame (type: Pandas DataFrame)
    :param pool: pool of primers already chosen (type: list of str)
    :returns: list of primers chosen and set of genomes covered by new primers (type: 
    list of str and set of str respectively)
    '''
    genomes = set() # empty set for new-covered genomes
    newPrimers = list() # empty list for new primers
    if not len(pool): # removal of all oligos which form dimers with primers in pool
        rdf = rdf.loc[index_nhd(pool, rdf.index)]
        fdf = fdf.loc[index_nhd(pool, fdf.index)]
    
    for i in range(len(fdf.index)): # searching the primer pair
        r = diff2(rdf, fdf, fdf.index[i]) # finding the difference between
        if not len(r): # if there in no pairs, then go to another row
            pass
        else:
            r = r.loc[index_nhd([fdf.index[i]], r.index)] # checking dimers 
            # between forward primer and its pairs
            if not len(r): # if there in no pairs, then go to another row
                pass
            else:
                newPrimers.append(fdf.index[i]) # adding first primer to the primer list
                for i in range(len(r.index)):
                    # adding new primer if it will extend the set of covered genomes 
                    if (genomes | set(r.loc[r.index[i]][r.loc[r.index[i]] > 0].index.tolist()) != genomes):
                        genomes = genomes | set(r.loc[r.index[i]][r.loc[r.index[i]] > 0].index.tolist())
                        newPrimers.append(r.index[i])
                break
    return newPrimers, genomes # returning llist of new primers and set of new genomes
                

def cover_genomes(dfr, dff, iterations=20):
    '''
    The function gets forward primer DataFrame, reverse primer DataFrame and number of iterations.
    It returns list of dataframes with primers and genomes.
    :param dfr: DataFrame of reverse candidate oligos and treir positions in genomes (type: Pandas DataFrame)
    :param dff: DataFrame of forward candidate oligos and treir positions in genomes (type: Pandas DataFrame)
    :param iterations: number of iterations, indirectly regulates number of primers in pool (type: int)
    :returns: list of Pandas DataFrames with information about chosen primers and amplified genomes 
              (type: list of Pandas DataFrames)
    '''
    # making copies of initial DataFrames
    df1 = dfr.copy()
    df2 = dff.copy()
    
    
    outs = list() # empty list for DataFrames
    genomes_all = set() # empty set for amplified genomes
    primers = list() # empty list for primers
    
    for i in range(iterations): # going through iterations
        # getting primers and genomes
        new_primers, genomes = make_pairs(df1, df2, primers)
        # adding to outs part of initial DataFrame with info about chosen primers and genomes
        outs.append(df2[genomes].loc[new_primers])
        
        # dropping all amplified genomes from reverse and forward tables
        df1.drop(list(genomes), axis=1, inplace=True)
        df2.drop(list(genomes), axis=1, inplace=True)

        # sorting oligos by number of matched genomes
        df1 = sort_by_nz(df1)
        df2 = sort_by_nz(df2)
        
        # updating list of chosen oligos and set of amlified genomes
        genomes_all |= genomes
        primers += new_primers
        
        # displaying numbers of chosen oligos and amplified genomes
        print('Iteration {}: {} primers to detect {} genomes.'.format(i + 1, len(primers), len(genomes_all)))
        
    return outs


if __name__ == "__main__":
    print('Reading the matrix...')
    df = pd.read_csv('aln_matrix.csv', index_col='Unnamed: 0') # opening matrix file
    print('Initial matrix shape: {}.'.format(df.shape))
    
    print('Filtering by Tm and homodimer presence...')
    
    # Вместо того, чтобы считать большие матрицы, а затем проверять решения на наличие димеров и на Tm,
    # можно сначала проверить все олигонуклеотиды на соответстетствие необходимым условиям (это гораздо быстрее, чем обрабатывать большие матрицы), 
    # a потом считать маленькие матрицы, которые в разы, а то и в десятки раз меньше исходной, причём дальше проверки уже будут ненужны,
    # останется разве что проверить совместимость разных праймеров для использования в одном пуле. 

    # filtering all oligos by Tm and heterodimers
    df = df.loc[list(filter(lambda x: (not ishet(x, x)) and (Tm(x) <= 64) and (Tm(x) >= 62), df.index))]
    print('Pre-filtered matrix shape: {}.'.format(df.shape))


    print('Starting calculations...')
    
    # actually calculations
    ress2 = cover_genomes(df, df)

    # concatenation of the data
    dfo = pd.concat(ress2).abs()
    dfo.to_excel('greedy_result_example.xlsx')

    print('Finished for 20-iterations pool!')
    print('Number of primers: {}.'.format(len(dfo.index))) # total time
    print('Number of amplified oligos: {}.'.format(len(dfo.columns))) # time per row
    time.sleep(1) # sleping 1 second
