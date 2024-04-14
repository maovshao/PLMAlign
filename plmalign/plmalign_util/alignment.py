
from typing import Union, List, Tuple, Dict

import numpy as np
import pandas as pd

from .numeric import fill_score_matrix, traceback_from_point_opt2

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


def plmalign_gather_all_paths(array: np.ndarray,
					norm: bool = True,
					mode: str = 'local',
					gap_extension: float = 1.0,
					with_scores: bool = False) -> List[np.ndarray]:
	'''
	calculate scoring matrix from input substitution matrix `array`
	find all Smith-Waterman-like paths from bottom and right edges of scoring matrix
	Args:
		array (np.ndarray): raw subtitution matrix aka densitymap
		norm_rows (bool, str): whether to normalize array per row or per array
		mode: (str) set global or local alignment procedure
		with_scores (bool): if True return score matrix
	Returns:
		list: list of all valid paths through scoring matrix
		np.ndarray: scoring matrix used
	'''
	assert isinstance(mode, str)
	assert mode in {'global', 'local'}
	assert isinstance(gap_extension, (int, float))
	
	if not isinstance(array, np.ndarray):
		array = array.numpy().astype(np.float32)
	if not isinstance(norm, (str, bool)):
		raise ValueError(f'norm_rows arg should be bool type, but given: {norm}')
	# standarize embedding
	if isinstance(norm, bool):
		if norm:
			arraynorm = (array - array.mean())/(array.std() + 1e-3)
		else:
			arraynorm = array.copy()
	score_matrix = fill_score_matrix(arraynorm, gap_extension = gap_extension, mode=mode)
	# get all edge indices for left and bottom
	# score_matrix shape array.shape + 1
	# local alignment mode
	if mode == 'local':
		indice = np.unravel_index(np.argmax(score_matrix, axis=None), score_matrix.shape)
		#indice = (indice[0], indice[1]) 
	# global alignment mode
	elif mode == 'global':
		indice = (score_matrix.shape[0] - 1, score_matrix.shape[1] - 1)
	path = traceback_from_point_opt2(score_matrix, indice, mode=mode)
	if with_scores:
		return (path, score_matrix)
	else:
		return path

def plmalign_search_paths(submatrix: np.ndarray,
		 path: Tuple[list],
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
	#assert isinstance(path, list)
	assert isinstance(mode, str)
	assert mode in {"local", "global"}
	assert isinstance(as_df, bool)

	if not np.issubsctype(submatrix, np.float32):
		submatrix = submatrix.astype(np.float32)
	spans_locations = dict()
	path = path - 1
	y, x = path[::-1, 0].ravel(), path[::-1, 1].ravel()
	spans = [(0, len(path))]
	ipath = 0
	if any(spans):
		for idx, (start, stop) in enumerate(spans):
			alnlen = stop - start
			y1, x1 = y[start:stop], x[start:stop]
			arr_values = submatrix[y1, x1]
			arr_indices = np.stack([y1, x1], axis=1)
			keyid = f'{ipath}_{idx}'
			spans_locations[keyid] = {
				'pathid': 0,
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