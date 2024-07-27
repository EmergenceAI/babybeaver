import torch, torch.nn as nn, numpy as np, time
from sklearn.cluster import KMeans
from scipy.stats import entropy
from collections import Counter
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score

def flatten(x): return x.view(list(x.shape[:2]) + [-1])

################################################################ Simple lap timer
def tformat(s): return f'{int(s//3600):3}:{int(s//60)%60:02}:{int(s)%60:02}'
class LapTimer:
    def __init__(self):
        self.last = self.start = time.time()
        self.phase = 0
    def lap(self, pname=''):
        self.phase += 1
        now = time.time()
        lap = now - self.last
        self.last = now
        dstr = f'TIMER-{self.phase-1}: Time for PHASE-{self.phase-1} {tformat(lap)}  Total: {tformat(now-self.start)}'
        print(f'\n{dstr}\n\nPHASE-{self.phase}: {pname}')

################################################################ Model stuff
# Encoder for single letter
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.feedFnorward = nn.Sequential(
            # nn.Linear(input_size,  hidden_size), nn.SiLU(),
            nn.Linear(input_size,  hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, latent_size), nn.Softmax(dim=1))
    def forward(self, x):
        return self.feedFnorward(x.view(-1, self.input_size))

################################################################ Single character EMNIST test
def image_test(encoder, emnist_data):
    train_x, train_y, test_x, test_y, _classes = emnist_data
    shift = 1
    encoder.eval()
    with torch.no_grad():
        train_out = encoder(train_x)
        test_out = encoder(test_x)
        # Shift all classes by 1 bc of zero-based indexing, but 1-based labels.
        train_acc = (shift+train_out.argmax(1) == train_y).float().mean()
        test_acc = (shift+test_out.argmax(1) == test_y).float().mean()
        print(f'Train acc: {train_acc:.4f}')
        print(f'Test  acc: {test_acc:.4f}')
    return train_acc.detach().cpu().item(), test_acc.detach().cpu().item()

def cheat_train(train_x, train_y, device):
    train_x = train_x.to(device)
    train_y = train_y.clone().to(device)
    train_y -= 1
    return train_x, train_y

def train_model_cheat(model, emnist_data, steps, verbose=False):
    train_x, train_y, test_x, test_y, _classes = emnist_data
    train_x, train_y = cheat_train(train_x, train_y, model.device)
    test_x, test_y = cheat_train(test_x, test_y, model.device)
    encoder = model.encoder
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    best_test_loss = float('inf')
    best_test_checkpoint = encoder.state_dict()
    for i in range(steps):
        optimizer.zero_grad()
        guesses = encoder(train_x) #pylint: disable=not-callable
        loss = torch.nn.functional.cross_entropy(guesses, train_y)
        loss.backward()
        optimizer.step()
        if (i % 100 == 0 or i >= steps-1):
            # Keep the weights for the best test model
            with torch.no_grad():
                test_guesses = encoder(test_x)
                test_loss = torch.nn.functional.cross_entropy(test_guesses, test_y)
                if test_loss < best_test_loss:
                    if verbose:
                        print(f'Step {i:5d}, test loss: {test_loss:.4f}')
                        image_test(encoder, emnist_data)
                    best_test_loss = test_loss
                    best_test_checkpoint = encoder.state_dict()
    # Restore the best test model
    encoder.load_state_dict(best_test_checkpoint)
    return best_test_loss

################################################################ Clustering
def cluster(batch, words, CHARS):
    # Convert word string to np array
    wordstr = ''.join(words)
    labels = np.array([ord(c) for c in wordstr])
    # Cluster batch
    kmeans = KMeans(n_clusters=26, random_state=0, init='k-means++').fit(batch)
    c2x = {c: [] for c in CHARS}
    for c, x in zip(labels, kmeans.labels_):
        c2x[chr(c)].append(kmeans.labels_[x])
    # Create a counter for each cluster and get the entropy of each
    counts = {c: Counter(c2x[c]) for c in c2x}
    entropies = {c: entropy(list(counts[c].values())) for c in counts}
    for c in sorted(entropies): print(f'Cluster {c}: Entropy {entropies[c]:.2f}')
    print('Clusters')
    for c in sorted(counts): print(f'Cluster {c}: {counts[c]}')

################################################################ Fnord it up, brah!
# Fnord detector (Hack: Handcoded classifier: Can train classifier if we run on a bunch of models.)
def get_fnord_probs(batch, model, FNORD, CHARS):
    model.eval()
    char2idx = {c: i for i, c in enumerate(CHARS)}
    fnordidx = [char2idx[c] for c in FNORD]
    # Get the fnord probabilities
    batch = flatten(batch)
    probs_fnords = model(batch) #pylint: disable=not-callable
    fnord_probs = torch.diagonal(probs_fnords[:,:,fnordidx], 0, 1, 2)
    fnord_probs = torch.sum(torch.log(fnord_probs), axis=1)
    return fnord_probs

# Have you detected a fnord lately?
def detect_fnords(batch, model, FNORD, log_fnord_thresh, CHARS):
    fnord_probs = get_fnord_probs(batch, model, FNORD, CHARS)
    return fnord_probs > log_fnord_thresh*len(FNORD)

def fnord_test(FNORD, fnords, normal, fnord_prior, top_model, CHARS):
    fnord_examples = len(fnords)
    ################ Test fnord detection.
    # Get other supervised scores, like F1 and accuracy
    log_probs_pos = get_fnord_probs(fnords, top_model, FNORD, CHARS).detach().cpu().numpy()
    log_probs_neg = get_fnord_probs(normal, top_model, FNORD, CHARS).detach().cpu().numpy()
    # Get roc_auc_score
    y_true = np.concatenate([np.ones(fnord_examples), np.zeros(fnord_examples)])
    y_scores = np.concatenate([log_probs_pos, log_probs_neg])
    log_thresh = np.sort(y_scores)[int(fnord_prior * len(y_scores))]
    # Re-exponentialize the scores
    y_scores = np.exp(y_scores)
    # Sort the y_scores and get the fnord_prior percentile
    y_pred = np.concatenate([log_probs_pos > log_thresh, log_probs_neg > log_thresh])
    log_fnord_thresh = log_thresh / len(FNORD)
    classespos = detect_fnords(fnords, top_model, FNORD, log_fnord_thresh, CHARS)
    classesneg = detect_fnords(normal, top_model, FNORD, log_fnord_thresh, CHARS)
    return (
        roc_auc_score(y_true, y_scores),
        average_precision_score(y_true, y_scores),
        f1_score(y_true, y_pred),
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        torch.sum(classespos).item()/fnord_examples,
        torch.sum(classesneg).item()/fnord_examples
    )
