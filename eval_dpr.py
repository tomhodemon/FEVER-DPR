import json
import os
import time
from tqdm import tqdm

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import faiss

import utils


def main(args):    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    d = 768
    indexer = faiss.IndexFlatIP(d)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    if args.lora:
        base_model = AutoModel.from_pretrained("bert-base-cased", add_pooling_layer=False)
        
        queryEncoder = PeftModel.from_pretrained(base_model, args.query_encoder).to(device)
        queryEncoder.eval()

        passageEncoder = PeftModel.from_pretrained(base_model, args.passage_encoder).to(device)
        passageEncoder.eval()

    else:
        queryEncoder = AutoModel.from_pretrained(args.query_encoder, add_pooling_layer=False).to(device)
        queryEncoder.eval()

        passageEncoder = AutoModel.from_pretrained(args.passage_encoder, add_pooling_layer=False).to(device)
        passageEncoder.eval()

    # load dataset
    eval_dataset = load_dataset("tomhodemon/fever_data", split=args.split)

    # indexes
    if args.indexes is None:
        index_fn = lambda batch: utils.index_fn(batch, tokenizer, passageEncoder, indexer, device)
        eval_dataset.map(index_fn, batched=True, batch_size=8)
        if args.save_indexes is not None:
            faiss.write_index(indexer, f"{args.save_indexes}.bin")
    else:
        indexer = faiss.read_index(args.indexes) 

    collate_fn = lambda batch: utils.collate_fn(batch, tokenizer)
    batch_size = 8
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    st = time.time()
    TP = 0
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            query_tensors = {key: val.to(device) for key, val in batch.items()}
            query_embeddings = queryEncoder(**query_tensors).last_hidden_state[:,0,:].detach().cpu().numpy()
        _, I = indexer.search(query_embeddings, k=args.k)
       
        for i, idxs in enumerate(I.tolist()):
            query_idx = (step*batch_size)+i
            target_idx = (step*batch_size)+query_idx
            """
            if not args.recall_only:
                data.append({
                    "query_idx": (step*batch_size)+i,
                    "retrieved_idxs": idxs,
                    "target_idx": target_idx
                })
            """
            
            TP += 1 if target_idx in idxs else 0

    N = len(eval_dataset)
    entry = {
            f"recall@{args.k}": TP/N,
            "runtime:": f"{(time.time()-st):.2f}s",
            "N (num_query)": N,
            "indexer.ntotal": indexer.ntotal,
            "device": device,
            "query_encoder_name": args.query_encoder,
            "passage_encoder_name": args.passage_encoder,
        }

    fname = f'{args.output_dir}/recall@k.jsonl'
    feeds = []
    if not os.path.isfile(fname):
        feeds.append(entry)
        with open(fname, mode='w') as f:
            f.write(json.dumps(feeds, indent=2))
    else:
        with open(fname, 'r') as f:
            feeds = json.load(f)
        feeds.append(entry)
        with open(fname, mode='w') as f:
            f.write(json.dumps(feeds, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--indexes", type=str) # default is None
    parser.add_argument("--save_indexes", type=str) # only if indexes is not None
    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--passage_encoder", type=str, required=True)
    parser.add_argument("--query_encoder", type=str, required=True)

    args = parser.parse_args()
    main(args)