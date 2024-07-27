import torch, numpy as np, random, pickle, string, os, itertools, pickle
from collections import Counter
from datasets import load_dataset
from torchvision import datasets, transforms
from libbeaver import flatten

################################################################ Data stuff
CHARS = string.ascii_lowercase
class CharToVec:
    def __init__(self, classes, y):
        self.letters = {}
        for i, d in enumerate(y):
            c = classes[d]
            if c not in self.letters: self.letters[c] = []
            self.letters[c].append(i)
    def __call__(self, c: str):
        return -1 if c not in self.letters else random.choice(self.letters[c])

def get_next_char(lang: str='en'):
    myiter = load_dataset('wikipedia', f'20220301.{lang}', split='train', trust_remote_code=True).iter(1)
    while True:
        for doc in myiter:
            for c in doc['text'][0]:
                c = c.lower()
                if c in CHARS: yield c

def get_test_data(dataset, device=None):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if dataset == 'EMNIST':
        trainset = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
        num_train = 100000
        classes = trainset.classes
        label_shift = 0
    elif dataset == 'CIFAR':
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        num_train = 10000
        classes = ['N/A', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        label_shift = 1
    else: assert False
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True, num_workers=2)
    inputs, labels = next(iter(trainloader))
    inputs = (inputs + 1) / 2 # scale inputs to 0-1
    labels += label_shift
    # Only keep the first 26 classes.
    inputs = inputs[labels < 26]
    labels = labels[labels < 26]
    train_x, test_x = inputs[:num_train], inputs[num_train:]
    train_y, test_y = labels[:num_train], labels[num_train:]
    if device is not None:
        train_x, train_y = train_x.to(device), train_y.to(device)
        test_x, test_y = test_x.to(device), test_y.to(device)
    return train_x, train_y, test_x, test_y, classes

def load_data(dataset, device=None):
    train_x, train_y, test_x, test_y, classes = get_test_data(dataset, device)
    chars = iter(get_next_char())
    c2x = CharToVec(classes, train_y)
    c2x_test = CharToVec(classes, test_y)
    return c2x, c2x_test, chars, train_x, test_x

# Return zeros if the character is not in a-z.  (A space or punctuation.)
def encode_word(word: str, c2x, train_x):
    idxs = [c2x(c) for c in word]
    zeros = torch.zeros(train_x.shape[1:]).to(train_x.device)
    return torch.stack([(train_x[i] if i >= 0 else zeros) for i in idxs])

# Gets a batch from the chars iterator.
def get_batch(chars, c2x, train_x, batch_size, wordlen):
    words = [''.join([next(chars) for _ in range(wordlen)]) for _ in range(batch_size)]
    return get_batch2(words, c2x, train_x), words

# Gets a batch of the chars in words..
def get_batch2(words, c2x, train_x):
    return torch.stack([encode_word(word, c2x, train_x) for word in words])

################################################################ The innate bigram model and fnord detector
def get_bigram_mat():
    pickle_file = './data/bigrams.pkl'
    if os.path.exists(pickle_file):
        bigrams = pickle.load(open(pickle_file, 'rb'))
    else:
        print('  Could not find bigrams pickle file.  Recalculating.  (This will take a while.)')
        bigrams = Counter()
        chars = get_next_char()
        last = next(chars)
        total = 0
        for c in chars:
            bigrams[''.join([last, c])] += 1
            last = c
            total += 1
            if total > 1_000_000_000: break
        try: pickle.dump(bigrams, open(pickle_file, 'wb'))
        except: print('  Could not pickle bigrams.') #pylint: disable=bare-except # noqa
    mat = np.zeros([len(CHARS)]*2, dtype=np.float32)
    for i, j in itertools.product(range(len(CHARS)), repeat=2):
        mat[i, j] = bigrams[''.join([CHARS[i], CHARS[j]])]
    mat /= np.sum(mat) # Normalize
    freqs = np.sum(mat, axis=0)
    # We want to make probs where probs[k,j] is the probability of k given j.
    probs = np.zeros([len(CHARS)]*2, dtype=np.float32)
    for j, k in itertools.product(range(len(CHARS)), repeat=2):
        probs[k, j] = mat[j, k] / np.sum(mat[j])
    return probs, mat, freqs

################################################################ Our main functions
def get_all_test_data(fnord, device, dataset, trigger_examples, fnord_examples):
    # Load from pickle if there.
    datafile = f'./data/testdata_{dataset}.pkl'
    if not os.path.exists(datafile):
        print(f'Warning: {datafile} does not exist. Creating it and loading test data.')
        # If not, give warning and create the data
        _c2x, c2x_test, chars, _train_x, test_x = load_data(dataset)
        # Try 100 random triggers and see if we can detect them.
        triggers = [] # [FNORD, 'these', 'the', 'thing', 'spatula', 'trash', 'waka waka', 'darg', 'the duck flies at midnight']
        for tlen in range(2, 12):
            for _ in range(10): triggers.append(''.join([random.choice(CHARS) for _ in range(tlen)])) # Random
            for _ in range(10): triggers.append(''.join([next(chars) for _ in range(tlen)])) # From chars
        fnordsnormals = {}
        for trigger in triggers:
            fnords = get_batch2([trigger]*(trigger_examples), c2x_test, test_x)
            normal, _ = get_batch(chars, c2x_test, test_x, trigger_examples, len(trigger))
            fnordsnormals[trigger] = (fnords, normal)
        fnords = get_batch2([fnord]*(fnord_examples), c2x_test, test_x)
        normal, _ = get_batch(chars, c2x_test, test_x, fnord_examples, len(fnord))
        image_data = get_test_data(dataset)
        # Pickle
        with open(datafile, 'wb') as f:
            pickle.dump((fnordsnormals, fnords, normal, image_data), f)
    else:
        with open(datafile, 'rb') as f:
            fnordsnormals, fnords, normal, image_data = pickle.load(f)
    # Put these on the device
    fnordsnormals = {k: (v[0].to(device), v[1].to(device)) for k, v in fnordsnormals.items()}
    fnords = fnords.to(device)
    normal = normal.to(device)
    image_data = [x.to(device) for x in image_data[:-1]] + [image_data[-1]]
    return fnordsnormals, fnords, normal, image_data

def get_train_eval_data(device, dataset, train_batch_size, eval_size):
    c2x, _c2x_test, chars, train_x, _test_x = load_data(dataset, device)
    eval_words = [CHARS]*eval_size
    eval_batch = get_batch2(eval_words, c2x, train_x)
    eval_batch = flatten(eval_batch.to(device))
    train_batch, train_words = get_batch(chars, c2x, train_x, train_batch_size, 2)
    train_batch = train_batch.to(device)
    train_batch = flatten(train_batch)
    return train_batch, train_words, eval_batch, eval_words
