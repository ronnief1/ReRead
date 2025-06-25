from torch import nn
from transformers import BertTokenizer
from model.bert import BertForMaskedLM
from torch.nn import functional as F
from TokenMasker import TokenMasker
import torch
import copy
import math
import numpy as np
from util_funcs import load_vision_backbone


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELU(nn.Module):
    def forward(self, input_):
        output = gelu(input_)
        return output


class Contra_head(nn.Module):
    def __init__(self, input_dim, contra_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, contra_dim, bias=False)
    def forward(self, cls_token):
        return self.linear(cls_token)


def pool_text_for_contra(feature):
    return feature[:,0]#.unsqueeze(1)

def pool_video_for_contra(feature, token_index):  #feature b ,n ,x ,c
    global_features = feature[:,token_index,:]#.unsqueeze(1) #CLS token is last one from features
    return global_features


class Match_head(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = GELU()
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.linear2 = nn.Linear(hidden_size, 2)
    def forward(self, cls_token):
        return self.linear2(self.layernorm(self.activation(self.linear1(cls_token))))


class RFM(nn.Module):
    def __init__(self,
                 vision_weights_path='path/to/model',
                 bert_weights='pretrained_weights/BERT/bert-base-uncased-crossattn',
                 tokenizer_weights='pretrained_weights/BERT/tokenizer',
                 temperature=0.07,
                 exp_temp=False,
                 loss_weights={loss: 1 for loss in ["mlm", "gm", "itm", "itc"]}
                ):
        super().__init__()

        self.vision_weights_path = vision_weights_path
        self.cls_token_index = 0

        self.vision_encoder = load_vision_backbone(self.vision_weights_path).to(device)
        self.vision_encoder.encoder_dim = 768

        self.multimodal_encoder = BertForMaskedLM.from_pretrained(bert_weights).to(device)
        self.multimodal_encoder.tokenizer = BertTokenizer.from_pretrained(tokenizer_weights)
        self.multimodal_encoder.tokenizer.cls_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.multimodal_encoder.tokenizer.bos_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.multimodal_encoder.tokenizer.eos_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.multimodal_encoder.tokenizer.pad_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        self.multimodal_encoder.tokenizer.mask_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        self.multimodal_encoder.tokenizer.itm_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.multimodal_encoder.tokenizer.mlm_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.multimodal_encoder.tokenizer.itc_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]

        self.multimodal_encoder.multimodal_dim = 768
        self.text_mask_token = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        self.text_masker = TokenMasker(mask_token = self.text_mask_token, range_start=106, range_end = 30522).to(device)

        self.contra_dim = 512
        self.contra_head_t = Contra_head(self.multimodal_encoder.multimodal_dim, self.contra_dim).to(device)
        self.contra_head_v = Contra_head(self.vision_encoder.encoder_dim, self.contra_dim).to(device)

        self.itm_head = Match_head(self.multimodal_encoder.multimodal_dim).to(device)

        self.mlm_weight = loss_weights['mlm']
        self.gm_weight = loss_weights['gm']
        self.itc_weight = loss_weights['itc']
        self.itm_weight = loss_weights['itm']

        print('MLM weight: ', self.mlm_weight)
        print('GM weight: ', self.gm_weight)
        print('ITM weight: ', self.itm_weight)
        print('ITC weight: ', self.itc_weight)

        self.temperature = temperature
        self.exp_temp = exp_temp

        if self.exp_temp:
            self.contra_temp = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature)).to(device)
            print('Using exponential ', str(self.temperature), ' as CL temperature')
        else:
            self.contra_temp = nn.Parameter(torch.tensor(self.temperature)).to(device)
            print('Using ', str(self.temperature), ' as CL temperature')


    def forward(self, samples):

        tokens = self.multimodal_encoder.tokenizer(
            samples['texts'],
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt"
        ).to(device)

        with torch.cuda.amp.autocast():
            model_features = self.vision_encoder.get_vision_features(samples['images']).to(device)

        mlm_loss = self.forward_mlm(tokens, model_features) * self.mlm_weight
        gm_loss = self.forward_gm(tokens, model_features) * self.gm_weight
        loss_itc = self.forward_itc(tokens, model_features) * self.itc_weight
        loss_itm = self.forward_itm(tokens, model_features) * self.itm_weight

        return (mlm_loss + gm_loss + loss_itc + loss_itm)


    def forward_mlm(self, tokens, model_features):
        text_tokens = copy.deepcopy(tokens)
        input_ids = text_tokens['input_ids']
        input_ids[:,0] = self.multimodal_encoder.tokenizer.mlm_token_id
        attention_mask = text_tokens['attention_mask']
        input_ids, txt_labels = self.text_masker(input_ids, 0.15)

        video_input = model_features

        output = self.multimodal_encoder(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                                    encoder_hidden_states=video_input.to(device), labels = txt_labels.to(device))

        return output.loss



    def forward_gm(self, tokens, model_features):
#        if compute_loss:
        text_tokens = copy.deepcopy(tokens)
        input_ids, attention_mask = text_tokens['input_ids'], text_tokens['attention_mask']
        input_ids[:,0] = self.multimodal_encoder.tokenizer.bos_token_id

        input_ids, txt_labels = self.text_masker(input_ids, 0.6)

        sample_num = [1]*model_features.shape[0]#batch['sample_num']
        video_input = model_features

        video_input_expand = []
        for i in range(video_input.shape[0]):
            video_input_expand.append( video_input[i:i+1].expand(sample_num[i],-1,-1))
        video_input = torch.cat(video_input_expand,dim=0)

        seq_len = attention_mask.shape[1]
        attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_len, -1).clone()
        attention_mask[:, : seq_len, : seq_len] = torch.tril(attention_mask[:, : seq_len, : seq_len])

        output = self.multimodal_encoder(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                                    encoder_hidden_states=video_input.to(device), labels = txt_labels.to(device))

        return output.loss

    def forward_retrieval(self, tokens, model_features):
        text_tokens = copy.deepcopy(tokens)

        input_ids = text_tokens.input_ids
        input_ids[:,0] = self.multimodal_encoder.tokenizer.itc_token_id
        attention_mask = text_tokens.attention_mask

        text_output = self.multimodal_encoder(input_ids = input_ids.to(device), attention_mask = attention_mask.to(device))
        feat_text = pool_text_for_contra(text_output.sequence_output)
        feat_text = self.contra_head_t(feat_text)
        feat_text =  F.normalize(feat_text, dim=-1)

        feat_img = pool_video_for_contra(model_features, self.cls_token_index)
        feat_img = self.contra_head_v(feat_img)
        feat_img = F.normalize(feat_img, dim=-1)

        sim_i2t = torch.matmul(feat_img, feat_text.permute(1,0))
        sim_t2i = torch.matmul(feat_text, feat_img.permute(1,0))

        if self.exp_temp:
            sim_i2t = sim_i2t * self.contra_temp.exp()
            sim_t2i = sim_t2i * self.contra_temp.exp()  # [batch_size, batch_size*num_gpu]
        else:
            sim_i2t = sim_i2t / self.contra_temp
            sim_t2i = sim_t2i / self.contra_temp  # [batch_size, batch_size*num_gpu]


        bs = feat_img.size(0)

        targets = torch.arange(sim_i2t.shape[0])

        loss_itc = (
            F.cross_entropy(sim_i2t.to(device), targets.to(device), label_smoothing=0.1)
            + F.cross_entropy(sim_t2i.to(device), targets.to(device), label_smoothing=0.1)
        ) / 2

        loss_itc = loss_itc * self.itc_weight

        input_ids = copy.deepcopy(input_ids)
        input_ids[:, 0] = self.multimodal_encoder.tokenizer.itm_token_id  # Set the ITM token

        # No need for ddp_allgather, we work with the tensors directly
        input_ids_collate = input_ids
        attention_mask_collate = attention_mask
        image_embeds_world = model_features

        # No gradient required for sampling
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
            weights_t2i.fill_diagonal_(0)  # Set diagonal to 0 to avoid positive matches
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
            weights_i2t.fill_diagonal_(0)

        # Hard negative mining
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()  # Sample a negative image index
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)  # Stack the negative image embeddings

        # Hard negative mining for text
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()  # Sample a negative text index
            text_ids_neg.append(input_ids_collate[neg_idx])
            text_atts_neg.append(attention_mask_collate[neg_idx])

        # Stack the negative text token IDs and attention masks
        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # Concatenate positives and negatives for text and video
        batch_size = image_embeds_neg.shape[0]
        input_ids = torch.cat((input_ids, input_ids, text_ids_neg), dim=0)
        attention_mask = torch.cat((attention_mask, attention_mask, text_atts_neg), dim=0)
        video_output = torch.cat((model_features, image_embeds_neg, model_features), dim=0)

        output = self.multimodal_encoder(input_ids=input_ids.to(device),
                                         attention_mask=attention_mask.to(device),
                                         encoder_hidden_states=video_output.to(device)).sequence_output
        # Create ground truth labels
        ground_truth = torch.zeros(batch_size * 3).long()#.cuda()  # Assign to GPU
        ground_truth[:batch_size] = 1  # First part is the positive match

        # Pass the output through the ITM head (two-layer MLP)
        logits = self.itm_head(output[:, 0])
        # Compute ITM loss with cross-entropy
        loss_itm = F.cross_entropy(logits.to(device), ground_truth.to(device))

        # Scale and store the loss
        loss_itm = loss_itm * self.itm_weight

        #return loss_dict
        return loss_itc, loss_itm


    def forward_itc(self, tokens, model_features):
        text_tokens = copy.deepcopy(tokens)

        input_ids = text_tokens.input_ids
        input_ids[:,0] = self.multimodal_encoder.tokenizer.itc_token_id
        attention_mask = text_tokens.attention_mask

        text_output = self.multimodal_encoder(input_ids = input_ids.to(device), attention_mask = attention_mask.to(device))
        feat_text = pool_text_for_contra(text_output.sequence_output)
        feat_text = self.contra_head_t(feat_text)
        feat_text =  F.normalize(feat_text, dim=-1)

        feat_img = pool_video_for_contra(model_features, self.cls_token_index)
        feat_img = self.contra_head_v(feat_img)
        feat_img = F.normalize(feat_img, dim=-1)

        sim_i2t = torch.matmul(feat_img, feat_text.permute(1,0))
        sim_t2i = torch.matmul(feat_text, feat_img.permute(1,0))

        if self.exp_temp:
            sim_i2t = sim_i2t * self.contra_temp.exp()
            sim_t2i = sim_t2i * self.contra_temp.exp()  # [batch_size, batch_size*num_gpu]
        else:
            sim_i2t = sim_i2t / self.contra_temp
            sim_t2i = sim_t2i / self.contra_temp  # [batch_size, batch_size*num_gpu]

        targets = torch.arange(sim_i2t.shape[0])

        loss_itc = (
            F.cross_entropy(sim_i2t.to(device), targets.to(device), label_smoothing=0.1)
            + F.cross_entropy(sim_t2i.to(device), targets.to(device), label_smoothing=0.1)
        ) / 2

        return loss_itc


    def forward_itm(self, tokens, model_features):
        text_tokens = copy.deepcopy(tokens)

        input_ids = text_tokens.input_ids
        input_ids[:,0] = self.multimodal_encoder.tokenizer.itc_token_id
        attention_mask = text_tokens.attention_mask

        text_output = self.multimodal_encoder(input_ids = input_ids.to(device), attention_mask = attention_mask.to(device))
        feat_text = pool_text_for_contra(text_output.sequence_output)
        feat_text = self.contra_head_t(feat_text)
        feat_text =  F.normalize(feat_text, dim=-1)

        feat_img = pool_video_for_contra(model_features, self.cls_token_index)
        feat_img = self.contra_head_v(feat_img)
        feat_img = F.normalize(feat_img, dim=-1)

        sim_i2t = torch.matmul(feat_img, feat_text.permute(1,0))
        sim_t2i = torch.matmul(feat_text, feat_img.permute(1,0))

        if self.exp_temp:
            sim_i2t = sim_i2t * self.contra_temp.exp()
            sim_t2i = sim_t2i * self.contra_temp.exp()  # [batch_size, batch_size*num_gpu]
        else:
            sim_i2t = sim_i2t / self.contra_temp
            sim_t2i = sim_t2i / self.contra_temp  # [batch_size, batch_size*num_gpu]


        bs = feat_img.size(0)

        # targets = torch.arange(sim_i2t.shape[0])

        # loss_itc = (
        #     F.cross_entropy(sim_i2t.to(device), targets.to(device), label_smoothing=0.1)
        #     + F.cross_entropy(sim_t2i.to(device), targets.to(device), label_smoothing=0.1)
        # ) / 2

        # loss_itc = loss_itc * self.itc_weight

        input_ids = copy.deepcopy(input_ids)
        input_ids[:, 0] = self.multimodal_encoder.tokenizer.itm_token_id  # Set the ITM token

        # No need for ddp_allgather, we work with the tensors directly
        input_ids_collate = input_ids
        attention_mask_collate = attention_mask
        image_embeds_world = model_features

        # No gradient required for sampling
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
            weights_t2i.fill_diagonal_(0)  # Set diagonal to 0 to avoid positive matches
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
            weights_i2t.fill_diagonal_(0)

        # Hard negative mining
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()  # Sample a negative image index
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)  # Stack the negative image embeddings

        # Hard negative mining for text
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()  # Sample a negative text index
            text_ids_neg.append(input_ids_collate[neg_idx])
            text_atts_neg.append(attention_mask_collate[neg_idx])

        # Stack the negative text token IDs and attention masks
        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # Concatenate positives and negatives for text and video
        batch_size = image_embeds_neg.shape[0]
        input_ids = torch.cat((input_ids, input_ids, text_ids_neg), dim=0)
        attention_mask = torch.cat((attention_mask, attention_mask, text_atts_neg), dim=0)
        video_output = torch.cat((model_features, image_embeds_neg, model_features), dim=0)

        output = self.multimodal_encoder(input_ids=input_ids.to(device),
                                         attention_mask=attention_mask.to(device),
                                         encoder_hidden_states=video_output.to(device)).sequence_output
        # Create ground truth labels
        ground_truth = torch.zeros(batch_size * 3).long()#.cuda()  # Assign to GPU
        ground_truth[:batch_size] = 1  # First part is the positive match

        # Pass the output through the ITM head (two-layer MLP)
        logits = self.itm_head(output[:, 0])
        # Compute ITM loss with cross-entropy
        loss_itm = F.cross_entropy(logits.to(device), ground_truth.to(device))

        return loss_itm

