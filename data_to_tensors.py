import pickle
import torch
from tqdm.notebook import tqdm

MAX_NUM_PATHS  = 200

class DatasetBuiled:
    '''
    The purpose of this class is to process a dataset's 
    each line and form 3 tensors of start_words, contexts,
    end_words. Lines are looks like 'method_name start_word, 
    context, end_word'
    '''
    def __init__(self, dataset_path, value_vocab, path_vocab):
        self.dataset_path = dataset_path
        self.value_vocab = value_vocab
        self.path_vocab = path_vocab

    def _data_processing_one_line(self, line):
        final_start = []
        final_path = []
        final_ends = []
        name, *tree = line.split(' ')
        for each_path in tree:
            if each_path != '' and each_path != '\n':
                temp_path = each_path.split(',')
                # in order not to add such path, which consist of words not in our dictionary(from dataset)
                if None not in [self.value_vocab.get(temp_path[0]), 
                                self.path_vocab.get(temp_path[1]), 
                                self.value_vocab.get(temp_path[2])]:

                  final_start.append(self.value_vocab[temp_path[0]])
                  final_path.append(self.path_vocab[temp_path[1]])
                  final_ends.append(self.value_vocab[temp_path[2]])

        final_start += [self.value_vocab['<pad>']] * (MAX_NUM_PATHS - len(final_start))
        final_path += [self.value_vocab['<pad>']] * (MAX_NUM_PATHS - len(final_path))
        final_ends += [self.value_vocab['<pad>']] * (MAX_NUM_PATHS - len(final_ends))
        return final_start, final_path, final_ends

    def _form_tensors(self):
        list_starts = []
        list_contexts = []
        list_ends = []
        with open(self.dataset_path, 'r') as file:

            for line in tqdm(file):
                        test_ = self._data_processing_one_line(line)
                        list_starts += [test_[0]]
                        list_contexts += [test_[1]]
                        list_ends += [test_[2]]

        tensor_starts = torch.LongTensor(list_starts)
        tensor_contexts = torch.LongTensor(list_contexts)
        tensor_ends = torch.LongTensor(list_ends)
        return tensor_starts, tensor_contexts, tensor_ends