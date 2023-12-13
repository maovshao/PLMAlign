import itertools

import pandas as pd
from typing import Union
import numpy as np
from tqdm import tqdm

from .plmalign_util.alignment import plmalign_gather_all_paths, plmalign_search_paths
from .plmblast_util.alignment import plmblast_gather_all_paths, plmblast_search_paths
from .util import filter_result_dataframe, dot_product, read_fasta, embedding_load, embedding_cos_similarity, make_parent_dir, draw_alignment
from .embedding_generate import prottrans_embedding_generate

class plmalign:
    '''
    main class for handling alignment extaction
    '''
    NORM: Union[bool, str] = True
    MODE: str = 'local'
    GAP_EXT: float = 1.0

    def __init__(self, *args, **kw_args):
        pass

    def embedding_to_span(self, X, Y, result_mode : str = 'results') -> pd.DataFrame:

        ### PLMAlign (dot)
        X = X.numpy()
        Y = Y.numpy()
        densitymap = dot_product(X, Y)

        densitymap = densitymap.T
        
        path = plmalign_gather_all_paths(densitymap,
                                norm=self.NORM,
                                mode=self.MODE,
                                gap_extension=self.GAP_EXT,
                                with_scores = True if result_mode == 'all' else False)
        if result_mode == 'all':
            scorematrix = path[1]
            path = path[0]
        results = plmalign_search_paths(densitymap,
                               path=path,
                               mode=self.MODE,
                               as_df=True)
        if result_mode == 'all':
            return (results, densitymap, path, scorematrix)
        else:
            return results

class plmblast:
    '''
    main class for handling alignment extaction
    '''
    MIN_SPAN_LEN: int = 20
    WINDOW_SIZE: int = 20
    NORM: Union[bool, str] = True
    BFACTOR: str = 'local'
    SIGMA_FACTOR: float = 1
    GAP_OPEN: float = 0.0
    GAP_EXT: float = 0.0
    FILTER_RESULTS: bool = False

    def __init__(self, *args, **kw_args):
        pass

    def embedding_to_span(self, X, Y, mode : str = 'results') -> pd.DataFrame:

        ### pLM-BLAST (cos)
        X = X.numpy()
        Y = Y.numpy()
        densitymap = embedding_cos_similarity(X, Y)

        densitymap = densitymap.T
        
        paths = plmblast_gather_all_paths(densitymap,
                                 norm=self.NORM,
                                 minlen=self.MIN_SPAN_LEN,
                                 bfactor=self.BFACTOR,
                                 gap_opening=self.GAP_OPEN,
                                 gap_extension=self.GAP_EXT,
                                 with_scores = True if mode == 'all' else False)
        if mode == 'all':
            scorematrix = paths[1]
            paths = paths[0]
        results = plmblast_search_paths(densitymap,
                               paths=paths,
                               window=self.WINDOW_SIZE,
                               min_span=self.MIN_SPAN_LEN,
                               sigma_factor=self.SIGMA_FACTOR,
                               mode=self.BFACTOR,
                               as_df=True)
        if mode == 'all':
            return (results, densitymap, paths, scorematrix)
        else:
            return results


    def full_compare(self, emb1: np.ndarray, emb2: np.ndarray,
                     idx: int = 0, file: str = 'source.fasta') -> pd.DataFrame:
        '''
        Args:
            emb1: (np.ndarray) sequence embedding [seqlen x embdim]
            emb2: (np.ndarray) sequence embedding [seqlen x embdim]
            idx: (int) identifier used when multiple function results are concatenated
            file: (str) embedding/sequence source file may be omitted
        Returns:
            data: (pd.DataFrame) frame with alignments and their scores
        '''
        res = self.embedding_to_span(emb1, emb2)
        if len(res) > 0:
            # add referece index to each hit
            res['i'] = idx
            res['dbfile'] = file
            # filter out redundant hits
            if self.FILTER_RESULTS:
                res = filter_result_dataframe(res, column='score')
        return res

def pairwise_align(embedding1, embedding2, seq1, seq2, mode, method = 'plmalign'):
    if method == 'plmalign':
        extr = plmalign()
        extr.MODE = mode
        results = extr.embedding_to_span(embedding2, embedding1)
    elif method == 'plmblast':
        extr = plmblast()
        extr.BFACTOR = mode
        extr.FILTER_RESULTS = True
        if mode == 'local':
            results = extr.full_compare(embedding2, embedding1)
            if len(results) == 0:
                extr.BFACTOR = 'global'
                results = extr.embedding_to_span(embedding2, embedding1)
                extr.BFACTOR = 'local'
        else:
            results = extr.embedding_to_span(embedding2, embedding1)
    else:
        assert method in {"plmalign", "plmblast"}

    # Print best alignment
    row = results.iloc[0]

    aln = draw_alignment(row.indices, seq1, seq2, output='str')

    return row['score'].item(), aln

def plmalign_pipeline(query_fasta, target_fasta, mode = 'global', query_embedding_path = None, target_embedding_path = None, search_result_setting = None, output_path = None, if_stdout = True):
    method = 'plmalign'
    print(f"Align with method: {method}")

    _, query_sequences = read_fasta(query_fasta)
    _, target_sequences = read_fasta(target_fasta)

    if (query_embedding_path == None):
        query_embeddings = prottrans_embedding_generate(query_fasta)
    else:
        query_embeddings = embedding_load(query_fasta, query_embedding_path)

    if (target_embedding_path == None):
        target_embeddings = prottrans_embedding_generate(target_fasta)
    else:
        target_embeddings = embedding_load(target_fasta, target_embedding_path)
    
    if (output_path != None):
        output_score = output_path + 'score'
        make_parent_dir(output_score)
        output_alignment = output_path + 'alignment'
        f1 = open(output_score, 'w')
        f2 = open(output_alignment, 'w')
        
        protein_pair_dict = {}
        for protein in query_sequences:
            protein_pair_dict[protein] = []
        output_score_sort = output_path + 'score_sort'

    if (search_result_setting == None):
        for single_query in tqdm(query_sequences, desc="Query"):
            for single_target in target_sequences:
                score, results = pairwise_align(query_embeddings[single_query], target_embeddings[single_target], query_sequences[single_query], target_sequences[single_target], mode, method = method)
                if if_stdout:
                    print(f"{single_query}\t{single_target}\t Score = {score}\n")
                    print(f"{single_query}\t{single_target}\n{results}\n")
                if (output_path != None):
                    f1.write(f"{single_query}\t{single_target}\t{score}\n")
                    f2.write(f"{single_query}\t{single_target}\n{results}\n\n")
                    protein_pair_dict[single_query].append((single_target, score))

    else:
        search_result_path = search_result_setting[0]
        top = search_result_setting[1]

        with open(search_result_path, "r") as f:
            pairs = f.readlines()

        for line in tqdm(pairs, desc="Search result"):
            single_query, single_target, similarity = line.strip().split()
            similarity = eval(similarity)

            if ((top != None) and (similarity<top)):
                continue

            score, results = pairwise_align(query_embeddings[single_query], target_embeddings[single_target], query_sequences[single_query], target_sequences[single_target], mode, method = method)
            if if_stdout:
                print(f"{single_query}\t{single_target}\t Score = {score}\n")
                print(f"{single_query}\t{single_target}\n{results}\n")
            if (output_path != None):
                f1.write(f"{single_query}\t{single_target}\t{score}\n")
                f2.write(f"{single_query}\t{single_target}\n{results}\n\n")
                protein_pair_dict[single_query].append((single_target, score))
    
    if (output_path != None):
        f1.close()
        f2.close()
    
        for query_protein in query_sequences:
            protein_pair_dict[query_protein] = sorted(protein_pair_dict[query_protein], key=lambda x:x[1], reverse=True)
        
        with open(output_score_sort, 'w') as f3:
            for query_protein in query_sequences:
                for pair in protein_pair_dict[query_protein]:
                    f3.write(f"{query_protein}\t{pair[0]}\t{pair[1]}\n")