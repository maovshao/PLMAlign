# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tqdm import tqdm
from pathlib import Path

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer

def esm_embedding_generate(fasta, embedding_path = None, esm_model_path = 'data/model/esm/esm1b_t33_650M_UR50S.pt', nogpu = False):
    esm_model, alphabet = pretrained.load_model_and_alphabet(esm_model_path)
    esm_model.eval()
    if isinstance(esm_model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )

    if torch.cuda.is_available() and not nogpu:
        esm_model = esm_model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(fasta)
    batches = dataset.get_batch_indices(16384, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
    )
    print(f"Read {fasta} with {len(dataset)} sequences")

    embedding_result_dic = {}
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and not nogpu:
               toks = toks.to(device="cuda", non_blocking=True)

            out = esm_model(toks, repr_layers=[33], return_contacts=False)["representations"][33]

            for i, label in enumerate(labels):
                #get mean embedding
                esm_embedding = out[i, 1 : len(strs[i]) + 1].clone().cpu()
                embedding_result_dic[label] = esm_embedding
                if embedding_path != None:
                    embedding_path = Path(embedding_path)
                    output_file = embedding_path / f"{label}.pt"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        esm_embedding,
                        output_file,
                    )
    
    return embedding_result_dic

from transformers import T5Tokenizer, T5EncoderModel

def prottrans_embedding_generate(fasta,
                embedding_path = None, 
                prottrans_model_path = "data/model/Rostlab/prot_t5_xl_uniref50/",
                nogpu = False,
                max_residues=4000, # number of cumulative residues per batch
                max_seq_len=1022, # max length after which we switch to single-sequence processing to avoid OOM
                max_batch=50 # max number of sequences per single batch
                ):
    def prottrans_get_t5_model(prottrans_model_path):
        if prottrans_model_path is not None:
            print("##########################")
            print("Loading cached model from: {}".format(prottrans_model_path))
            print("##########################")
        model = T5EncoderModel.from_pretrained(prottrans_model_path)

        model = model.to(device)
        model = model.eval()
        tokenizer = T5Tokenizer.from_pretrained(prottrans_model_path, do_lower_case=False)
        return model, tokenizer

    def prottrans_read_fasta( fasta_path ):
        '''
            Reads in fasta file containing multiple sequences.
            Returns dictionary of holding multiple sequences or only single 
            sequence, depending on input file.
        '''
        sequences = dict()
        with open( fasta_path, 'r' ) as fasta_f:
            for line in fasta_f:
                # get uniprot ID from header and create new entry
                if line.startswith('>'):
                    uniprot_id = line.replace('>', '').strip()
                    sequences[ uniprot_id ] = ''
                else:
                    # repl. all whie-space chars and join seqs spanning multiple lines
                    sequences[ uniprot_id ] += ''.join( line.split() ).upper().replace("-","") # drop gaps and cast to upper-case
                    
        return sequences

    device = torch.device('cuda:0' if torch.cuda.is_available() and not nogpu else 'cpu')
    print("Using device: {}".format(device))

    seq_dict = dict()
    emb_dict = dict()

    # Read in fasta
    seq_dict = prottrans_read_fasta(fasta)
    model, vocab = prottrans_get_t5_model(prottrans_model_path)

    print('########################################')
    print('Total number of sequences: {}'.format(len(seq_dict)))

    avg_length = sum([ len(seq) for _, seq in seq_dict.items()]) / len(seq_dict)
    n_long     = sum([ 1 for _, seq in seq_dict.items() if len(seq)>max_seq_len])
    seq_dict   = sorted( seq_dict.items(), key=lambda kv: len( seq_dict[kv[0]] ), reverse=True )
    
    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))
    
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(tqdm(seq_dict),1):
        seq = seq.replace('U','X').replace('Z','X').replace('O','X')
        seq = seq[:max_seq_len]
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = vocab.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
            
            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={}). Try lowering batch size. ".format(pdb_id, seq_len) +
                      "If single sequence processing does not work, you need more vRAM to process your protein.")
                continue
            
            # batch-size x seq_len x embedding_dim
            # extra token is added at the end of the seq
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = min(seq_lens[batch_idx], max_seq_len)

                emb = embedding_repr.last_hidden_state[batch_idx,:s_len].clone().cpu()
            
                if len(emb_dict) == 0:
                    print("Embedded protein {} with length {} to emb. of shape: {}".format(
                        identifier, s_len, emb.shape))

                emb_dict[ identifier ] = emb
                if embedding_path != None:
                    embedding_path = Path(embedding_path)
                    output_file = embedding_path / f"{identifier}.pt"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        emb,
                        output_file,
                    )

    print('\n############# STATS #############')
    print('Total number of embeddings: {}'.format(len(emb_dict)))
    return emb_dict
