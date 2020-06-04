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
               path2idx=dict()):
    super().__init__()

    self.values_vocab_size = values_vocab_size
    self.paths_vocab_size = paths_vocab_size
    self.val_embedding_dim = val_embedding_dim
    self.path_embedding_dim = path_embedding_dim
    self.dropout_rate = dropout_rate
    self.embedding_dim = embedding_dim
    self.labels_num = labels_num
    self.path2idx = path2idx

    ## 1. Embeddings
    self.values_embedding = nn.Embedding(self.values_vocab_size, self.val_embedding_dim)
    self.paths_embedding = nn.Embedding(self.paths_vocab_size, self.path_embedding_dim)

    ## 2. DropOut + tanh(Fully-connected layer) for combined context vectors
    self.DropOut = nn.Dropout(self.dropout_rate)
    self.linear = nn.Linear(self.path_embedding_dim + 2 * self.val_embedding_dim, self.embedding_dim, bias = False)

    ## 3. Attention vector a
    self.a = nn.Parameter(torch.randn(1, self.embedding_dim))
    context_size = self.path_embedding_dim + 2 * self.val_embedding_dim
    #vocab_size = self.values_vocab_size, self.paths_vocab_size
    configuration = BertConfig(type_vocab_size=1, vocab_size=self.values_vocab_size, hidden_size=self.embedding_dim, num_attention_heads=8, intermediate_size=context_size)
    self.bert = BertModel(configuration)
    print (self.bert.modules)

    ## 4. Prediction
    self.output_linear = nn.Linear(self.embedding_dim, self.labels_num, bias = False)
    print (self.embedding_dim, self.labels_num)

    self.neg_INF = - 2 * 10**10

  def forward(self, starts, paths, ends, labels):

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

    ## 3. DropOut + Fully-connected layer into 'Combinied context vectors'
    context_vec = self.DropOut(context_vec)
    comb_context_vec = torch.tanh(self.linear(context_vec)) 

    ## 4. Attention mechanism
    mask = (starts > 1).float() ## if 1 then it is pad and we don't pay attention to it
    print (mask.shape)
    lin_mul = torch.matmul(comb_context_vec, self.a.T)

    attention_weights = F.softmax(torch.mul(lin_mul, mask.view(lin_mul.size())) + (1 - mask.view(lin_mul.size())) * self.neg_INF, dim = 1)
    code_vector = torch.sum(torch.mul(comb_context_vec, attention_weights), dim = 1)


    ## 5. Prediction
    output = self.output_linear(code_vector)
    print (code_vector.shape, output.shape)
    # output = F.softmax(output, dim = 1)
    #label_ids = torch.cat((starts, paths, ends), dim=1)
    b_output = self.bert(attention_mask=mask, inputs_embeds=comb_context_vec)
    print (b_output[0].shape, b_output[1].shape)
    #return b_output

    return code_vector, output
