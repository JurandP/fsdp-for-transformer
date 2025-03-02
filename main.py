import os
import argparse
import torch
import mlflow
import mlflow.pytorch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import GPT2TokenizerFast
import functools
from torch.nn.attention import SDPBackend
from datasets import load_from_disk
from torch.distributed.checkpoint import FileSystemReader
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import OrderedDict
import torch.distributed.checkpoint as dist_cp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import time
from datetime import datetime

############## MODEL PART ###############

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        positions = (
            torch.arange(seq_len, dtype=torch.long, device=x.device)
            .unsqueeze(0)
            .expand_as(x)
        )
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)
        embeddings = token_embeddings + position_embeddings
        return embeddings


class AttentionLayer(nn.Module):
    def __init__(
        self,
        dmodel,
        heads,
    ):
        super(AttentionLayer, self).__init__()
        self.ln = nn.LayerNorm(dmodel)
        self.heads = heads
        self.input_projection = nn.Linear(dmodel, 3 * dmodel, bias=False)
        self.output_projection = nn.Linear(dmodel, dmodel, bias=False)

    def forward(self, x, attention_mask):
        x = self.ln(x)
        projected = self.input_projection(x)
        batch, seq_len = x.shape[:-1]
        q_chunk, k_chunk, v_chunk = torch.chunk(projected, chunks=3, dim=-1)
        query = q_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        key = k_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        value = v_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)

        with torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH,]):
            attention_output = F.scaled_dot_product_attention(query=query, key=key, value=value, attn_mask=attention_mask, is_causal=True,)
        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))

        return output

def FeedForward(dmodel):
    return nn.Sequential(
        OrderedDict([
            ("ff_layernorm", nn.LayerNorm(dmodel)),
            ("pre_relu", nn.Linear(dmodel, 4 * dmodel, bias=True,),),
            ("relu", nn.ReLU()),
            ("post_relu", nn.Linear(4 * dmodel, dmodel, bias=True,),),]))

class Block(nn.Module):

    def __init__(self, dmodel, heads,):
        super().__init__()
        self.attention_layer = AttentionLayer(dmodel, heads)
        self.feed_forward_layer = FeedForward(dmodel)

    def forward(self, x, attention_mask):
        out_attention = self.attention_layer(x, attention_mask)
        x = x + out_attention

        out_feed_forward = self.feed_forward_layer(x)
        x = x + out_feed_forward
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_layer = EmbeddingLayer(
            config.vocab_size, config.d_model, config.max_len
        )
        self.blocks = nn.ModuleList(
            [Block(config.d_model, config.num_heads) for _ in range(config.num_layers)]
        )

        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        output = self.embedding_layer(input_ids)

        for block in self.blocks:
            output = block(output, attention_mask)

        output = self.head(output)
        return output


#########################################

def setup():
    # initialize the process group
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()

def setup_model(args):
    model = Transformer(args)
    return model


def collate_tokenize(tokenizer, sequence_length, data):
    text_batch = [element["text"] for element in data]
    tokenized = tokenizer(
        text_batch,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=sequence_length + 1,
    )
    input_ids = tokenized["input_ids"]
    tokenized["input_ids"] = input_ids[:, :-1]
    tokenized["target_ids"] = input_ids[:, 1:]
    tokenized["attention_mask"] = tokenized["attention_mask"][:, :-1]
    return tokenized

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))))

    return LambdaLR(optimizer, lr_lambda)

def get_dataloader(
    batch_size,
    sequence_length,
    split="train",
    buffer_size=10000,
    seed=42,
    num_workers=2,
    world_size=1,
    rank=0,
):
    if split == "train":
        hf_dataset = load_from_disk("./datasets/c4/train")
    else:
        hf_dataset = load_from_disk("./datasets/c4/validation")


    sampler = DistributedSampler(hf_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=seed)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataloader = DataLoader(
        hf_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_tokenize, tokenizer, sequence_length),
        sampler=sampler,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    
    return dataloader, sampler

def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"Current date and time of run = {date_of_run}")
    return date_of_run

def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / (1024 * 1024)
    metric_num = round(metric_num, ndigits=4)
    return metric_num

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def validation(model, valid_dataloader, local_rank, validation_steps):
    model.eval()
    fsdp_loss = torch.zeros(1).to(local_rank)
    valid_losses = []
    
    with torch.no_grad():
        for _, batch in zip(range(validation_steps), valid_dataloader):
            input_ids = batch["input_ids"].to(local_rank)
            target_ids = batch["target_ids"].to(local_rank)
            attention_mask = batch["attention_mask"]
            outputs = model(input_ids)
            mask_loss = F.cross_entropy(
                outputs.flatten(0, -2),
                target_ids.reshape(-1).long(),
                reduction="none",
            )
            mask_loss = mask_loss[attention_mask.reshape(-1) == 1]
            loss = mask_loss.mean().item()
            valid_losses.append(loss)
            mean_valid_loss = sum(valid_losses) / validation_steps
            fsdp_loss[0] = mean_valid_loss 
        
    with torch.no_grad():
        dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
        
    fsdp_loss[0] = fsdp_loss[0] / dist.get_world_size()    
    
    return fsdp_loss[0]


def fsdp_main(args):

    model = setup_model(args)

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    setup()
    
    main_worker = ( local_rank == 0 and rank == 0)
    
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Block,
        },
    )
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD #SHARD_GRAD_OP
    torch.cuda.set_device(local_rank)

    mp_policy = MixedPrecision(
      param_dtype=torch.bfloat16,
      # Gradient communication precision.
      reduce_dtype=torch.bfloat16,
      # Buffer precision.
      buffer_dtype=torch.bfloat16,
    )

    # model is on CPU before input to FSDP
    model = FSDP(model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device())

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Set up cosine schedule with warmup
    num_warmup_steps = int(0.01 * args.train_steps)  # 1% of training steps for warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, args.train_steps)
    
    if args.load_model and main_worker:

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = {"model": model.state_dict()}
            dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader=dist_cp.FileSystemReader(args.load_folder),
            )
            model.load_state_dict(state_dict["model"])

            optim_state = load_sharded_optimizer_state_dict(
                model_state_dict=state_dict["model"],
                optimizer_key="optim",
                storage_reader=dist_cp.FileSystemReader(args.load_folder),
            )

            flattened_osd = FSDP.optim_state_dict_to_load(
                model, optimizer, optim_state["optim"]
            )
            optimizer.load_state_dict(flattened_osd)

            scheduler_state = torch.load(os.path.join(args.load_folder, "scheduler.pth"))
            scheduler.load_state_dict(scheduler_state)
        
    # Start MLflow experiment
    run = mlflow.start_run()
    mlflow.log_params(vars(args))  # Log all configuration parameters as tags

    if args.track_memory:
        mem_alloc_tracker = []
        mem_reserved_tracker = []

    train_dataloader, train_sampler = get_dataloader(
        batch_size=args.batch_size,
        sequence_length=args.seq_length,
        split="train",
        seed=args.seed,
        world_size=world_size,
        rank=rank)
    
    validation_dataloader, _ = get_dataloader(
        batch_size=args.batch_size,
        sequence_length=args.seq_length,
        split="validation",
        seed=args.seed,
        world_size=world_size,
        rank=rank)

    starting_time = time.time()

    fsdp_loss = torch.zeros(1).to(local_rank)
    validation_steps = int(1e06 // (args.batch_size * args.seq_length))

    if train_sampler:
        train_sampler.set_epoch(1)
        
    for i, batch in zip(range(args.train_steps), train_dataloader):
        model.train()
        input_ids = batch["input_ids"].to(local_rank)
        target_ids = batch["target_ids"].to(local_rank)
        attention_mask = batch["attention_mask"]
        optimizer.zero_grad()
        outputs = model(input_ids)
        
        mask_loss = F.cross_entropy(
            outputs.flatten(0, -2),
            target_ids.reshape(-1).long(),
            reduction="none",
        )
        mask_loss = mask_loss[attention_mask.reshape(-1) == 1]
        loss = mask_loss.mean()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        fsdp_loss[0] = loss.item()
        
        with torch.no_grad():
            dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
        
        fsdp_loss = fsdp_loss / dist.get_world_size()    
        
        if i % args.log_train_loss_freq == 0 and main_worker:
            
            # train_accuracy = fsdp_loss[0] / fsdp_loss[1]
            print(f"Step:{i}, Train Loss:{loss.item()}")
            mlflow.log_metric("train_loss", loss.item(), step=i) 
            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metric("learning_rate", current_lr, step=i)
        
            
        if args.track_memory:
            mem_allocated = format_metrics_to_gb(torch.cuda.memory_allocated())
            mem_reserved = format_metrics_to_gb(torch.cuda.memory_reserved())
            mem_alloc_tracker.append((mem_allocated, local_rank))
            mem_reserved_tracker.append((mem_reserved, local_rank))

            mlflow.log_metric(f"memory_allocated_gpu:{local_rank}_worker_{rank}", mem_allocated, step=i)
            mlflow.log_metric(f"memory_reserved_gpu:{local_rank}_worker_{rank}", mem_reserved, step=i) 

        if i % args.log_valid_loss_freq == 0:
            valid_loss = validation(model, validation_dataloader, local_rank, validation_steps)
        
        if i % args.log_valid_loss_freq == 0 and main_worker:
            print(f"Validation loss:{valid_loss}")
            mlflow.log_metric("valid_loss", valid_loss, step=i)

    final_valid_loss = validation(model, validation_dataloader, local_rank, validation_steps)
            
    if main_worker:
        print(f"Final validation loss:{final_valid_loss}")         
        mlflow.log_param("final_valid_loss", final_valid_loss)

        total_time = str(time.time() - starting_time)
        mlflow.log_param("training_time", total_time)
        
        num_param = count_parameters(model) * dist.get_world_size() # For FSDP, parameters are sharded between workers
        mlflow.log_param("number_of_parameters", num_param)
        print(f"This model has {num_param} parameters")
        
    if args.save_model and main_worker:

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": model.state_dict(),
                "optim": FSDP.optim_state_dict(model, optimizer),
            }

            dist_cp.save_state_dict(
                state_dict=state_dict,
                storage_writer=dist_cp.FileSystemWriter(args.saving_folder),
            )
            torch.save(scheduler.state_dict(), os.path.join(args.saving_folder, "scheduler.pth"))
            
        print("Model, optimizer, and scheduler saved successfully.")


    mlflow.end_run()
    dist.barrier()
    cleanup()

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='FSDP transformer training')
    parser.add_argument('--train_steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 4)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--track_memory', action='store_true', default=False, help='track the gpu memory')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--load_model', action='store_true', default=False, help='For Loading model from ./checkpoints')
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--max_len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimensionality')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--seq_length', type=int, default=256, help='Sequence length')
    parser.add_argument('--log_train_loss_freq', type=int, default=100, help='Frequency of logging training loss')
    parser.add_argument('--log_valid_loss_freq', type=int, default=100, help='Frequency of logging validation loss')
    parser.add_argument('--saving_folder', type=str, default="./checkpoints", help='Learning rate')
    parser.add_argument('--load_folder', type=str, default="./checkpoints", help='Learning rate')
    
    
    args = parser.parse_args()
    
    torch.cuda.empty_cache()

    torch.manual_seed(args.seed)

    fsdp_main(args)