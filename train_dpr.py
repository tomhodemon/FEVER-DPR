import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import transformers
from transformers import BertModel, BertTokenizer
from datasets import load_dataset
from peft import get_peft_model
from huggingface_hub import login

from tqdm import tqdm
import utils

transformers.logging.set_verbosity_error()

# tensorboard writer
writer = SummaryWriter()

class BiEncoder(nn.Module):
    def __init__(self, model_name, lora_config, p_dropout=0.1):
        super(BiEncoder, self).__init__()
        self.queryEncoder = BertModel.from_pretrained(model_name, add_pooling_layer=False)
        self.passageEncoder = BertModel.from_pretrained(model_name, add_pooling_layer=False)
        if lora_config is not None:
            self.queryEncoder = get_peft_model(self.queryEncoder, lora_config)
            self.queryEncoder.print_trainable_parameters()

            self.passageEncoder = get_peft_model(self.queryEncoder, lora_config)
            self.passageEncoder.print_trainable_parameters()
            
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, query_tensors, passage_tensors):
        query_embeddings = self.queryEncoder(**query_tensors).last_hidden_state[:,0,:] # returns [CLS] representations
        passage_embeddings = self.passageEncoder(**passage_tensors).last_hidden_state[:,0,:] # returns [CLS] representations

        query_embeddings = self.dropout(query_embeddings)
        passage_embeddings = self.dropout(passage_embeddings)

        return query_embeddings, passage_embeddings

    @staticmethod
    def compute_loss(scores, labels):
        probs = F.log_softmax(scores, dim=1)
        return F.nll_loss(probs, labels)
    
class FEVERDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super(FEVERDataset, self).__init__()
        self.data = load_dataset("tomhodemon/fever_data_dpr", split=split)
        print(f"Loaded {split} dataset:")
        print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]
    
    @staticmethod
    def collate_fn(batch, tokenizer):
    
        # prepare query tensors
        queries = [utils.normalize_query(item['query']) for item in batch]
        query_tensors = tokenizer(queries, truncation=True, padding=True, max_length=256, return_tensors='pt')

        # prepare passage tensors
        positive_passages = [item['positive_passage'] for item in batch]
        hard_negative_passages = [item['hard_negative_passage'] for item in batch]

        # in-batch negative
        passages = positive_passages + hard_negative_passages
        passages = {
            'titles': [p['title'] for p in passages],
            'passages': [p['passage'] for p in passages]
        }

        passage_tensors = tokenizer(passages['titles'], passages['passages'], truncation=True, padding=True, max_length=256, return_tensors='pt')

        return query_tensors, passage_tensors

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    for batch in dataloader:
        query_tensors, passage_tensors = batch
        query_tensors = {key: val.to(device) for key, val in query_tensors.items()}
        passage_tensors = {key: val.to(device) for key, val in passage_tensors.items()}

        query_embeddings, passage_embeddings = model(query_tensors, passage_tensors)

        score = torch.matmul(query_embeddings, passage_embeddings.permute(1, 0))
        labels = torch.arange(query_embeddings.size(0)).to(score.device)

        loss =  BiEncoder.compute_loss(score, labels)

        total_loss+=loss.item()

    total_loss /= len(dataloader)
    model.train()
    return total_loss

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.hf_token is not None:
        try:
            login(args.hf_token)
            args.push_to_hub = True
        except:
            print("Failed to login to HuggingFace Hub. Please check your token.")
            continue_running = input("Do you want to continue running the code? (Models will not be pushed to HF Hub after training) (y/n): ")
            if continue_running.lower() != 'y':
                print("Aborting...")
                return
            else:
                args.push_to_hub = False
       
    # tokenize & model
    tokenizer = BertTokenizer.from_pretrained(args.base_model_name)

    if args.lora:
        lora_config = utils.get_lora_config(args.lora_rank, args.lora_alpha, args.lora_dropout)
    else:
        lora_config = None
    
    bi_encoder = BiEncoder(args.base_model_name, lora_config).to(device)
    bi_encoder.train()

    # dataloaders
    collate_fn = lambda batch: FEVERDataset.collate_fn(batch, tokenizer)

    train_dataset = FEVERDataset("train")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    if args.do_eval:
        eval_dataset = FEVERDataset("validation")
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # optimizer & scheduler
    optimizer = torch.optim.Adam(bi_encoder.parameters(), lr=args.lr)
    scheduler = utils.get_schedule_linear(optimizer, args.warmup_steps, total_training_steps=args.max_steps)

    global_step = 0

    # training loop
    for epoch in range(args.epochs):
        for batch in (pbar := tqdm(train_dataloader, position=0, leave=True)):
            global_step += 1
            query_tensors, passage_tensors = batch
            query_tensors = {key: val.to(device) for key, val in query_tensors.items()}
            passage_tensors = {key: val.to(device) for key, val in passage_tensors.items()}

            optimizer.zero_grad()

            query_embeddings, passage_embeddings = bi_encoder(query_tensors, passage_tensors)

            score = torch.matmul(query_embeddings, passage_embeddings.permute(1, 0))
            labels = torch.arange(query_embeddings.size(0)).to(score.device)

            loss = BiEncoder.compute_loss(score, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if global_step % args.logging_steps == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix({'loss': loss.item()})

            if (global_step % args.eval_steps == 0) and args.do_eval:
                eval_loss = evaluate(bi_encoder, eval_dataloader, device)
                writer.add_scalar('Loss/eval', eval_loss, global_step)

            if global_step >= args.max_steps:
                break

    if args.push_to_hub:
        query_encoder_name = f"fever_query_encoder_{args.batch_size}_{args.max_steps}"
        print(f"Pushing to HF Hub as {query_encoder_name}")
        bi_encoder.queryEncoder.push_to_hub(query_encoder_name)

        passage_encoder_name = f"fever_passage_encoder_{args.batch_size}_{args.max_steps}"
        print(f"Pushing to HF Hub as {passage_encoder_name}")
        bi_encoder.passageEncoder.push_to_hub(passage_encoder_name)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

     # training arguments
    parser.add_argument('--baxse_model_name', type=str, default='bert-base-cased')
    parser.add_argument('--max_steps', type=int, default=6000)
    parser.add_argument('--logging_steps', type=int, default=20)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument('--hf_token', type=str) # if not None, models will be pushed to HF Hub

    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=10e-5)  

    # LoRA
    parser.add_argument('--lora', action="store_true")
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    args = parser.parse_args()
    
    main(args)