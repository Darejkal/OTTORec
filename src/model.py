import numpy as np
import torch

from src.customlog import CustomLogger
from src.evaluator import calculate_ranks, mean_metric, pointwise_mrr, pointwise_recall


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class DynamicPositionEmbedding(torch.nn.Module):

    def __init__(self, max_len, dimension):
        super(DynamicPositionEmbedding, self).__init__()
        self.max_len = max_len
        self.embedding = torch.nn.Embedding(max_len, dimension)
        self.pos_indices = torch.arange(0, self.max_len, dtype=torch.int)
        self.register_buffer('pos_indices_const', self.pos_indices)

    def forward(self, x, device='cpu'):
        seq_len = x.shape[1]
        return self.embedding(self.pos_indices_const[-seq_len:]) + x
def multiply_head_with_embedding(prediction_head, embeddings):
    return prediction_head.matmul(embeddings.transpose(-1, -2))
class ImprovisedSasrec(torch.nn.Module):
    def __init__(self, item_num,max_len,hidden_size,dropout_rate,num_layers,sampling_style,device="cpu",share_embeddings=True,topk_sampling=False,topk_sampling_k=1000):
        super(ImprovisedSasrec, self).__init__()
        self.hidden_size=hidden_size
        self.item_num = item_num
        self.share_embeddings=share_embeddings
        self.sampling_style=sampling_style
        self.topk_sampling=topk_sampling
        self.topk_sampling_k=topk_sampling_k
        self.merge_attn_mask = True
        self.device=device
        self.future_mask = torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1)
        self.register_buffer('future_mask_const', self.future_mask)
        self.register_buffer('seq_diag_const', ~torch.diag(torch.ones(max_len, dtype=torch.bool)))
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(item_num + 1, hidden_size, padding_idx=0)
        if share_embeddings:
            self.output_emb = self.item_emb
        self.pos_emb = DynamicPositionEmbedding(max_len,hidden_size)
        self.input_dropout = torch.nn.Dropout(p=dropout_rate)
        self.last_layernorm = torch.nn.LayerNorm(hidden_size)
        encoder_layer=torch.nn.TransformerEncoderLayer(d_model=hidden_size,
                                                   nhead=1,
                                                   dim_feedforward=hidden_size,
                                                   dropout=dropout_rate,
                                                   batch_first=True,
                                                   norm_first=True)
        self.encoder=torch.nn.TransformerEncoder(encoder_layer,num_layers,self.last_layernorm)
        # self.final_activation = torch.nn.ELU(0.5)
        self.final_activation = torch.nn.Identity()
            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()
        for _, param in self.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass # just ignore those failed init layers
    def merge_attn_masks(self, padding_mask):
        """
        padding_mask: 0 if padded and 1 if comes from the source sequence
        Returns a mask of size (batch,maxseq,maxseq) where True means masked (not allowed to attend) and False means otherwise.
        """
        batch_size = padding_mask.shape[0]
        seq_len = padding_mask.shape[1]

        if not self.merge_attn_mask:
            return self.future_mask_const[:seq_len, :seq_len]

        padding_mask_broadcast = ~padding_mask.bool().unsqueeze(1)
        future_masks = torch.tile(self.future_mask_const[:seq_len, :seq_len], (batch_size, 1, 1))
        merged_masks = torch.logical_or(padding_mask_broadcast, future_masks)
        # Always allow self-attention to prevent NaN loss
        # See: https://github.com/pytorch/pytorch/issues/41508
        diag_masks = torch.tile(self.seq_diag_const[:seq_len, :seq_len], (batch_size, 1, 1))
        return torch.logical_and(diag_masks, merged_masks)
    def forward(self, positives, mask): # for training   
        """
        item_indices: positives
        mask: padding mask of 0 and 1
        returns attention_head
        """ 
        att_mask = self.merge_attn_masks(mask)    
        items = self.item_emb(positives)
        x = items * torch.sqrt(torch.tensor(self.hidden_size))
        x = self.pos_emb(x)
        prediction_head = self.encoder(self.input_dropout(x), att_mask)
        return prediction_head
    def train_step(self, batch, iteration,logger:CustomLogger,optimizer,criterion:torch.nn.modules.loss._Loss):
        prediction_head = self.forward(batch["positives"],batch["mask"])

        pos = multiply_head_with_embedding(prediction_head.unsqueeze(-2),
                                                   self.output_emb(batch["labels"]).unsqueeze(-2)).squeeze(-1)

        if self.sampling_style == "eventwise":
            uniform_negative_logits = multiply_head_with_embedding(prediction_head.unsqueeze(-2),
                                                                self.output_emb(batch["uniform_negatives"])).squeeze(-2)
        else:
            uniform_negative_logits = multiply_head_with_embedding(prediction_head, self.output_emb(batch["uniform_negatives"]))

        in_batch_negative_logits = multiply_head_with_embedding(prediction_head, self.output_emb(batch["in_batch_negatives"]))
        neg = torch.concat([uniform_negative_logits, in_batch_negative_logits], dim=-1)
        if self.topk_sampling:
            neg, _ = torch.topk(neg, k=self.topk_sampling_k, dim=-1)
        pos_scores, neg_scores = self.final_activation(pos), self.final_activation(neg)
        loss_mask=torch.where(pos!=0)
        loss=criterion(pos_scores[loss_mask],torch.ones(pos_scores.shape, device=self.device)[loss_mask],)
        loss+=criterion(neg_scores[loss_mask],torch.ones(neg_scores.shape, device=self.device)[loss_mask],)

        logger.log("ITERATION",f"i: {iteration}, train_loss: {loss}", )
        loss.backward()
        optimizer.step()

    def validate_step(self, batch, iteration,logger:CustomLogger,criterion:torch.nn.modules.loss._Loss):
        prediction_head = self.forward(batch["positives"], batch["mask"])
        # loss:
        pos = multiply_head_with_embedding(prediction_head.unsqueeze(-2),
                                        self.output_emb(batch["labels"]).unsqueeze(-2)).squeeze(-1)
        if self.sampling_style == "eventwise":
            uniform_negative_logits = multiply_head_with_embedding(prediction_head.unsqueeze(-2),
                                                                self.output_emb(batch["uniform_negatives"])).squeeze(-2)
        else:
            uniform_negative_logits = multiply_head_with_embedding(prediction_head, self.output_emb(batch["uniform_negatives"]))
        in_batch_negative_logits = multiply_head_with_embedding(prediction_head, self.output_emb(batch["in_batch_negatives"]))
        neg = torch.concat([uniform_negative_logits, in_batch_negative_logits], dim=-1)
        if self.topk_sampling:
            neg, _ = torch.topk(neg, k=self.topk_sampling_k, dim=-1)
        pos_scores, neg_scores = self.final_activation(pos), self.final_activation(neg)
        loss_mask=torch.where(pos!=0)
        loss=criterion(pos_scores[loss_mask],torch.ones(pos_scores.shape, device=self.device)[loss_mask],)
        loss+=criterion(neg_scores[loss_mask],torch.ones(neg_scores.shape, device=self.device)[loss_mask],)
        # score:
        cut_offs = torch.tensor([5, 10, 20], device=self.device)
        recalls,mrrs=[],[]
        for t in range(prediction_head.shape[1]):
            mask = batch['mask'][:, t]
            positives = batch['labels'][:, t]
            logits = multiply_head_with_embedding(prediction_head[:, t], self.output_emb.weight)
            logits[:, 0] = -torch.inf  # set score for padding item to -inf
            ranks = calculate_ranks(logits, positives, cut_offs)
            pw_rec = pointwise_recall(ranks, cut_offs, mask)
            recalls.append(pw_rec.squeeze(dim=1))
            pw_mrr = pointwise_mrr(ranks, cut_offs, mask)
            mrrs.append(pw_mrr.squeeze(dim=1))
        pw_rec = torch.stack(recalls, dim=2)
        pw_mrr = torch.stack(mrrs, dim=2)
        recall,mrr= mean_metric(pw_rec, batch["mask"]), mean_metric(pw_mrr, batch["mask"])
        recall_sum=0
        mrr_sum=0
        count=0
        for i, k in enumerate(cut_offs.tolist()):
            recall_sum+=recall[i]
            mrr_sum+=mrr[i]
            logger.log(f"EVALUATE_{iteration}",f'recall_cutoff_{k}= {recall[i]}',True )
            logger.log(f"EVALUATE_{iteration}",f'mrr_cutoff_{k}={mrr[i]}',True)
            count+=1
        logger.log(f"EVALUATE_LOSS_{iteration}",f"loss={loss}")