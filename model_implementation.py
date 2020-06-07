import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class code2vec_model(nn.Module):
  """
  self.values_vocab_size - set of values of AST terminals that were observed during training
  self.paths_vocab_size - set of AST paths
  self.val_embedding_dim - size of embedding vector for values
  self.path_embedding_dim - size of embedding vector for values
  """
  def __init__(self, 
               val_embedding_dim = 128,
               path_embedding_dim = 128,
               dropout_rate = 0.25,
               embedding_dim = 128,
               values_vocab_size = 0,
               paths_vocab_size = 0,
               labels_num = 0,
               bert = False,
               bert_params = None):
    super().__init__()

    self.values_vocab_size = values_vocab_size
    self.paths_vocab_size = paths_vocab_size
    self.val_embedding_dim = val_embedding_dim
    self.path_embedding_dim = path_embedding_dim
    self.dropout_rate = dropout_rate
    self.embedding_dim = embedding_dim
    self.labels_num = labels_num
    self.bert = bert
    
    ## 1. Embeddings
    self.values_embedding = nn.Embedding(self.values_vocab_size, self.val_embedding_dim)
    self.paths_embedding = nn.Embedding(self.paths_vocab_size, self.path_embedding_dim)

    ## 2. DropOut + tanh(Fully-connected layer) for combined context vectors
    self.DropOut = nn.Dropout(self.dropout_rate)
    self.linear = nn.Linear(self.path_embedding_dim + 2 * self.val_embedding_dim, self.embedding_dim, bias = False)

    ## 3. Bert or attention vector a
    if bert and bert_params != None:
      num_attention_heads = bert_params['num_attention_heads']
      num_transformer_layers = bert_params['num_transformer_layers']
      intermediate_size = bert_params['intermediate_size']
      self.scale = torch.nn.Parameter(torch.sqrt(torch.tensor(self.embedding_dim, dtype=torch.float32)), requires_grad=False)
      max_len = 512
      self.pos_embedding = nn.Embedding(max_len, self.embedding_dim)
      configuration = BertConfig(type_vocab_size=1, vocab_size=self.labels_num, 
                                 hidden_size=self.embedding_dim, num_attention_heads=num_attention_heads, 
                                 num_hidden_layers=num_transformer_layers, intermediate_size=intermediate_size, 
                                 hidden_dropout_prob=dropout_rate, attention_probs_dropout_prob=dropout_rate, max_position_embeddings=max_len)
                                 
      self.bert = BertModel(configuration)
    else:
      self.a = nn.Parameter(torch.randn(1, self.embedding_dim))
    ## 4. Prediction
    self.output_linear = nn.Linear(self.embedding_dim, self.labels_num, bias = False)

    self.neg_INF = - 2 * 10**10

  def forward(self, starts, paths, ends):
    """
    input for starts,paths,ends - [[],[],[]...[]] - N_paths * BATCH_SIZE
    We form the indexed vocab of left_nodes, paths, right_nodes
    starts, paths, ends - lists of INDEXES of left_nodes, paths, right_nodes
    """
    
    ## 1. Embeddings
    start_embedding = self.values_embedding(starts)
    path_embedding = self.paths_embedding(paths)
    end_embedding = self.values_embedding(ends)

    ## 2. Concatecation of 3 vectors 
    context_vec = torch.cat((start_embedding, path_embedding, end_embedding), dim=2)
    
    ## 3. Attention mechanism
    mask = (starts > 1).float() ## if 1 then it is pad and we don't pay attention to it
    ## 4. DropOut + Fully-connected layer into 'Combinied context vectors'
    comb_context_vec = torch.tanh(self.linear(context_vec))
    
    if self.bert:
      bs, seq_len = comb_context_vec.shape[:2]
      token_embeds = self.scale * comb_context_vec
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      positions = torch.arange(seq_len).to(device).repeat(bs).view(bs, seq_len)
      pos_embeds = self.pos_embedding(positions)
      sum_embeds = torch.add(token_embeds, pos_embeds)
      context_vec = self.DropOut(sum_embeds)
      _, code_vector = self.bert(attention_mask=mask, inputs_embeds=context_vec)
    else:
      comb_context_vec = self.DropOut(comb_context_vec)
      lin_mul = torch.matmul(comb_context_vec, self.a.T)
      attention_weights = F.softmax(torch.mul(lin_mul, mask.view(lin_mul.size())) + (1 - mask.view(lin_mul.size())) * self.neg_INF, dim = 1)
      code_vector = torch.sum(torch.mul(comb_context_vec, attention_weights), dim = 1)
    ## 5. Prediction
    output = self.output_linear(code_vector)
    
    return code_vector, output
