import regex as re

STOP_TOKEN = '<|endoftext|>'
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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