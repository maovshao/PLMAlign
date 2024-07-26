
# Copyright (c) 2022 Laboratory of Structural Bioinformatics
#
# This file is modified from [https://github.com/labstructbioinf/pLM-BLAST].
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.

from typing import Union, List, Tuple, Dict

import numpy as np
import pandas as pd

from .numeric import fill_score_matrix, traceback_from_point_opt2, move_mean, find_alignment_span

def get_borderline(a: np.array, cutoff_h: int = 10, cutoff_w: int = 10) -> np.ndarray:
	'''
	extract all possible border indices (down, right) for given 2D matrix
	for example: \n
		A A A A A X\n
		A A A A A X\n
		A A A A A X\n
		A A A A A X\n
		A A A A A X\n
		X X X X X X\n
	\n
	result will contain indices of `X` values starting from upper right to lower left
	Args:
		a (np.ndarray):
		cutoff_h (int): control how far stay from edges - the nearer the edge the shorter diagonal for first dimension
		cutoff_w (int): control how far stay from edges - the nearer the edge the shorter diagonal for second dimension
	Returns:
		np.ndarray: border coordinates with shape of [len, 2] 
	'''
	# width aka bottom
	height, width = a.shape
	height -= 1; width -= 1
	# clip values		

	if height < cutoff_h:
		hstart = 0
	else:
		hstart = cutoff_h

	if width < cutoff_w:
		bstart = 0
	else:
		bstart = cutoff_w
	# arange with add syntetic dimension
	# height + 1 is here for diagonal
	hindices = np.arange(hstart, height+1)[:, None]
	# add new axis
	hindices = np.repeat(hindices, 2, axis=1)
	hindices[:, 1] = width

	# same operations for bottom line
	# but in reverted order
	bindices = np.arange(bstart, width)[::-1, None]
	# add new axis
	bindices = np.repeat(bindices, 2, axis=1)
	bindices[:, 0] = height
	
	borderline = np.vstack((hindices, bindices))
	return borderline

def border_argmaxpool(array: np.ndarray,
					cutoff: int = 10,
					factor: int = 2) -> np.ndarray:
	"""
	Get border indices of an array satysfing cutoff and factor conditions.

	Args:
		array (np.ndarray): embedding-based scoring matrix.
		cutoff (int): parameter to control border cutoff.
		factor (int): stride-like control of indices returned similar to path[::factor].

	Returns:
		(np.ndarray) path indices

	"""
	assert factor >= 1
	assert cutoff >= 0
	assert isinstance(factor, int)
	assert isinstance(cutoff, int)
	assert array.ndim == 2
	# case when short embeddings are given
	cutoff_h = cutoff if cutoff < array.shape[0] else 0
	cutoffh_w = cutoff if cutoff < array.shape[1] else 0
		
	boderindices = get_borderline(array, cutoff_h=cutoff_h, cutoff_w=cutoffh_w)
	if factor > 1:
		y, x = boderindices[:, 0], boderindices[:, 1]
		bordevals = array[y, x]
		num_values = bordevals.shape[0]	
		# make num_values divisible by `factor` 
		num_values = (num_values - (num_values % factor))
		# arange shape (num_values//factor, factor)
		# argmax over 1 axis is desired index over pool 
		arange2d = np.arange(0, num_values).reshape(-1, factor)
		arange2d_idx = np.arange(0, num_values, factor, dtype=np.int32)
		borderargmax = bordevals[arange2d].argmax(1)
		# add push factor so values  in range (0, factor) are translated
		# into (0, num_values)
		borderargmax += arange2d_idx
		return boderindices[borderargmax, :]
	else:
		return boderindices


def plmblast_gather_all_paths(array: np.ndarray,
					minlen: int = 10,
					norm: bool = True,
					bfactor: Union[int, str] = 1,
					gap_opening: float = 0,
					gap_extension: float = 0,
					with_scores: bool = False) -> List[np.ndarray]:
	'''
	calculate scoring matrix from input substitution matrix `array`
	find all Smith-Waterman-like paths from bottom and right edges of scoring matrix
	Args:
		array (np.ndarray): raw subtitution matrix aka densitymap
		norm_rows (bool, str): whether to normalize array per row or per array
		bfactor (int): use argmax pooling when extracting borders, bigger values will improve performence but may lower accuracy
		with_scores (bool): if True return score matrix
	Returns:
		list: list of all valid paths through scoring matrix
		np.ndarray: scoring matrix used
	'''
	
	if not isinstance(array, np.ndarray):
		array = array.numpy().astype(np.float32)
	if not isinstance(norm, (str, bool)):
		raise ValueError(f'norm_rows arg should be bool type, but given: {norm}')
	if not isinstance(bfactor, (str, int)):
		raise TypeError(f'bfactor should be int/str but given: {type(bfactor)}')
	# standarize embedding
	if isinstance(norm, bool):
		if norm:
			arraynorm = (array - array.mean())/(array.std() + 1e-3)
		else:
			arraynorm = array.copy()
	# set local or global alignment mode
	if bfactor == 'global':
		mode = 'global'
	else:
		mode = 'local'
	score_matrix = fill_score_matrix(arraynorm, gap_penalty=gap_opening, mode=mode)
	# get all edge indices for left and bottom
	# score_matrix shape array.shape + 1
	# local alignment mode
	if bfactor == 'local':
		indices = border_argmaxpool(score_matrix, cutoff=minlen, factor=1)
	# global alignment mode
	elif bfactor == 'global':
		indices = [(score_matrix.shape[0] - 1, score_matrix.shape[1] - 1)]
	paths = list()
	for ind in indices:
		path = traceback_from_point_opt2(score_matrix, ind, gap_opening=gap_opening)
		paths.append(path)
	if with_scores:
		return (paths, score_matrix)
	else:
		return paths

def plmblast_search_paths(submatrix: np.ndarray,
		 paths: Tuple[list, list],
		 window: int = 10,
		 min_span: int = 20,
		 sigma_factor: float = 1.0,
		 mode: str = 'local',
		 as_df: bool = False) -> Union[Dict[str, Dict], pd.DataFrame]:
	'''
	iterate over all paths and search for routes matching alignmnet criteria
	Args:
		submatrix: (np.ndarray) density matrix
		paths: (list) list of paths to scan
		window: (int) size of moving average window
		min_span: (int) minimal length of alignment to collect
		sigma_factor: (float) standard deviation threshold
		as_df: (bool) when True, instead of dictionary dataframe is returned
	Returns:
		record: (dict) alignment paths
	'''
	assert isinstance(submatrix, np.ndarray)
	assert isinstance(paths, list)
	assert isinstance(window, int) and window > 0
	assert isinstance(min_span, int) and min_span > 0
	assert isinstance(sigma_factor, (int, float))
	assert mode in {"local", "global"}
	assert isinstance(as_df, bool)

	AVG_EMBEDDING_STD = 0.1

	min_span = max(min_span, window)
	if not np.issubsctype(submatrix, np.float32):
		submatrix = submatrix.astype(np.float32)
	arr_sigma = submatrix.std()
	# force sigma to be not greater then average std of embeddings
	# also not too small
	arr_sigma = max(arr_sigma, AVG_EMBEDDING_STD)
	path_threshold = sigma_factor*arr_sigma
	spans_locations = dict()
	# iterate over all paths
	for ipath, path in enumerate(paths):
		# remove one index push
		diag_ind = path - 1
		if diag_ind.size < min_span:
			continue
		# revert indices and and split them into x, y
		y, x = diag_ind[::-1, 0].ravel(), diag_ind[::-1, 1].ravel()
		pathvals = submatrix[y, x].ravel()
		if mode == 'local':
			# smooth values in local mode
			if window != 1:
				line_mean = move_mean(pathvals, window)
			else:
				line_mean = pathvals
			spans = find_alignment_span(means=line_mean,
										mthreshold=path_threshold,
										minlen=min_span)
		else:
			spans = [(0, len(path))]
		# check if there is non empty alignment
		if any(spans):
			for idx, (start, stop) in enumerate(spans):
				alnlen = stop - start
				if alnlen < min_span:
					continue
				y1, x1 = y[start:stop-1], x[start:stop-1]
				arr_values = submatrix[y1, x1]
				'''
				if arr_values.mean() < path_threshold:
					print(arr_values)
					print(line_mean[start:stop])
					raise ValueError('array values are wrong', arr_values.mean(), path_threshold)
				'''
				arr_indices = np.stack([y1, x1], axis=1)
				keyid = f'{ipath}_{idx}'
				spans_locations[keyid] = {
					'pathid': ipath,
					'spanid': idx,
					'span_start': start,
					'span_end': stop,
					'indices': arr_indices,
					'score': arr_values.mean(),
					"len": alnlen,
					"mode": mode
				}
	if as_df:
		return pd.DataFrame(spans_locations.values())
	else:
		return spans_locations