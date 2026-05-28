import regex as re
from collections.abc import Iterable, Iterator
import json 

STOP_TOKEN = '<|endoftext|>'
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


import regex as re
from collections.abc import Iterable, Iterator
import json 

STOP_TOKEN = '<|endoftext|>'
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, vocab: dict[int: bytes], merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens is not None else []
        self.bytes_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

        for token in self.special_tokens:
            if token.encode('utf-8') not in self.vocab.values():
                self.vocab[len(vocab)] = token.encode('utf-8')

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab = {}
        merges = []

        with open(vocab_filepath, 'r') as file:
            vocab_tmp = json.load(file)
            vocab = {token_id : token_bytes_str.encode('utf-8') for token_bytes_str, token_id in vocab_tmp.items()}

        with open(merges_filepath, 'r') as file:
            merges = [tuple(line.split()) for line in file]

        return cls(vocab, merges, special_tokens.sort(key=len))
        
    def encode(self, text: str) -> list[int]:
        ids: [int] = []

        if len(self.special_tokens) > 0:
            pattern = '|'.join(map(re.escape, self.special_tokens))
            chunks = re.split(f'({pattern})', text)
        else:
            chunks = [text]

        all_encoded_pretokens = []

        for chunk in chunks:
            if chunk == '' or chunk is None: 
                continue
            if chunk in self.special_tokens:
                ids.append(self.bytes_to_id[chunk.encode('utf-8')])
                continue

            pretokens = re.findall(PAT, chunk)

            for pretoken in pretokens:
                encoded_bytes = pretoken.encode('utf-8')
                atoms = [bytes([b]) for b in encoded_bytes]
                atoms = self.apply_merges(atoms, self.merges)
                ids.extend([self.bytes_to_id[a] for a in atoms])
        return ids

    def apply_merges(self, atoms: list[bytes], merges: list[tuple]):
        
        for merge in merges:
            new_atoms = []
            i = 0

            while i < len(atoms):
                if i < len(atoms) - 1 and merge[0] == atoms[i] and merge[1] == atoms[i + 1]:
                    new_atoms.append(atoms[i] + atoms[i + 1])
                    i += 2
                else:
                    new_atoms.append(atoms[i])
                    i += 1
            atoms = new_atoms
        return atoms

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass

    def decode(self, ids: list[int]) -> str:
        final_bytes = bytearray()
        for id in ids:
            final_bytes.extend(self.vocab[id])
        return final_bytes.decode('utf-8', errors='replace')
        

def get_pretoken_counts(encoded_pretokens: list[tuple]):
    counts = {}
    for token in encoded_pretokens:
        counts[token] = counts.get(token, 0) + 1

    return counts

def get_pair_counts(pretoken_counts):
    pair_counts = {}
    for token, count in pretoken_counts.items():
        if len(token) < 2:
            continue
        i = 0
        j = 1
        while j < len(token):
            pair = (token[i], token[j])
            pair_counts[pair] = (pair_counts.get(pair, 0) + count)
            i += 1
            j += 1
                
    return pair_counts
        

def find_best_pair(pair_counts, vocab):
    max_count = 0
    best_pair = None

    for pair, count in pair_counts.items():
        if count == max_count and best_pair is not None:
            current_token = (vocab[pair[0]],  vocab[pair[1]])
            best_token = (vocab[best_pair[0]], vocab[best_pair[1]])
            if current_token > best_token:
                best_pair = pair
                max_count = count
        elif count > max_count:
            best_pair = pair
            max_count = count   
    return best_pair

def apply_merge(pretoken_counts, best_pair, merges, vocab):
    new_id = len(vocab)
    vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

    def get_new_tuple(pretoken_tuple, best_pair, vocab, new_id):
        i = 0
        new_tuple = ()

        while i < len(pretoken_tuple):
            if i < len(pretoken_tuple) - 1 and (pretoken_tuple[i], pretoken_tuple[i + 1]) == best_pair:
                new_tuple += (new_id,)
                i += 2
            else:
                new_tuple += (pretoken_tuple[i],)
                i += 1
        return new_tuple
                
    new_pretoken_counts = {}
    for pretoken, counts in pretoken_counts.items():
        key = get_new_tuple(pretoken, best_pair, vocab, new_id) 
        new_pretoken_counts[key] = new_pretoken_counts.get(key, 0) + counts
    merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
    return new_pretoken_counts


def read_input_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

def train_bpe(input_path: str, vocab_size = 260, special_tokens: list[str] = [STOP_TOKEN]):    

    text = read_input_file(input_path)

    pretoken_counts = {}
    merges = []

    pattern = '|'.join(map(re.escape, special_tokens))
    chunks = re.split(pattern, text)
    
    vocab = dict(zip([i for i in range(256)], [bytes([i]) for i in range(256)]))

    for token in special_tokens:
        vocab[len(vocab)] = bytes(token.encode('utf-8'))
    
    all_encoded_pretokens = []

    for chunk in chunks:
        pretokens = re.findall(PAT, chunk)
        encoded_pretokens = [tuple(pretoken.encode('utf-8')) for pretoken in pretokens]
        all_encoded_pretokens.extend(encoded_pretokens)

    pretoken_counts = get_pretoken_counts(all_encoded_pretokens)
    while len(vocab) < vocab_size:
        pair_counts = get_pair_counts(pretoken_counts)
        best_pair = find_best_pair(pair_counts, vocab)
        if best_pair is None:
            break
        pretoken_counts = apply_merge(pretoken_counts, best_pair, merges, vocab)

    return vocab, merges