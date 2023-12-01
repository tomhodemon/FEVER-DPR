import torch
from peft import LoraConfig

def normalize_query(query):
    query = query.replace("’", "'")
    return query

# from https://github.com/facebookresearch/DPR/blob/main/dpr/utils/model_utils.py#L106
def get_schedule_linear(
    optimizer,
    warmup_steps,
    total_training_steps,
    steps_shift=0,
    last_epoch=-1,
):

    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        current_step += steps_shift
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            1e-7,
            float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lora_config(lora_rank, lora_alpha, lora_dropout):
    return LoraConfig(
            target_modules=["query", "value"],
            inference_mode=False, 
            r=lora_rank, 
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none")

@torch.no_grad()
def index_fn(batch, tokenizer, encoder, indexer, device):
        positive_passages = batch['positive_passage']
        hard_negative_passages = batch['hard_negative_passage']

        passages = positive_passages + hard_negative_passages
        passages = {
            'titles': [p['title'] for p in passages],
            'passages': [p['text'] for p in passages]
        }

        passage_tensors = tokenizer(passages['titles'], passages['passages'], truncation=True, padding=True, max_length=256, return_tensors='pt')
        passage_tensors = {key: val.to(device) for key, val in passage_tensors.items()}
        passage_embeddings = encoder(**passage_tensors).last_hidden_state[:,0,:].detach().cpu().numpy()
        indexer.add(passage_embeddings)

def collate_fn(batch, tokenizer):
    
        # prepare query tensors
        queries = [normalize_query(item['query']) for item in batch]
        query_tensors = tokenizer(queries, truncation=True, padding=True, max_length=256, return_tensors='pt')
        return query_tensors