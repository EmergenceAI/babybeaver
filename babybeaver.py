# Innate fear of a fnord
import torch, torch.nn as nn, numpy as np, random, os, functools, argparse, builtins, csv
from tqdm import tqdm
from libbeaver import Encoder, image_test, LapTimer, train_model_cheat, cluster, fnord_test
from libdata import get_all_test_data, get_train_eval_data, CHARS, get_bigram_mat
print = functools.partial(builtins.print, flush=True) # Always flush print. #pylint: disable=redefined-builtin

################################################################ Encoders
def pred_loss(xenc, xpred): # Probability that I correctly predict the next letter.
    # Add one to avoid divide by zero.
    # The +1 is to treat this as a Dirichlet / Beta distribution with a uniform prior.
    # Thus, we get the expected value of this distribution instead of the maximum likelihood value.
    newbins = 1. + torch.sum(xenc, axis=0)
    newbins /= torch.sum(newbins)
    return torch.mean(-torch.log(torch.sum(xpred * xenc / newbins, axis=1)))

class WordEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, bigrams, mat, char_freqs, device, lossarity, losstype):
        super().__init__()
        self.num_chars = bigrams.shape[0]
        self.encoder = Encoder(input_size, hidden_size, self.num_chars)
        self.bigrams = torch.tensor(bigrams, device=device)
        self.mat = torch.tensor(mat, device=device)
        self.char_freqs = torch.tensor(char_freqs, device=device)
        self.lossarity = lossarity
        self.losstype = losstype
        self.device = device
    def forward(self, x):
        batch_size, word_length = x.shape[:2] # x is batch_size, word_length, input_size
        x = x.view([word_length * batch_size] + list(x.shape[2:]))
        return self.encoder(x).view(-1, word_length, self.num_chars)
    def loss_char_freqs(self, xenc):
        # See if the batch freqs match the char_freqs
        dist = torch.mean(xenc.reshape(-1, self.num_chars), axis=0)
        return torch.sum(dist * torch.log(dist / self.char_freqs)) # KL divergence...
    def loss(self, xenc):
        if self.lossarity == 'bigram':
            if self.losstype == 'kldivergence': # Bigram KL
                # Get the bigram probabilities from xenc, which is shape batch_size, 2, num_chars
                a = xenc[:,0] # batch_size, num_chars
                b = xenc[:,1] # batch_size, num_chars
                bigrams2 = torch.einsum('bi,bj->ij', a, b) # batch_size, num_chars, num_chars
                bigrams2 /= torch.sum(bigrams2) # Normalize
                p = bigrams2.view(-1)
                q = self.mat.view(-1)
                loss = torch.sum(p * torch.log(p / q)) # KL divergence
            else: # Bigram contrastive
                assert self.losstype == 'contrastive'
                xpred = torch.einsum('bi,ji->bj', xenc[:,0], self.bigrams)
                loss = pred_loss(xenc[:,1], xpred)
        else:
            assert self.lossarity == 'unigram'
            if self.losstype == 'kldivergence': # Unigram KL
                loss = self.loss_char_freqs(xenc)
            else: # Unigram contrastive
                assert self.losstype == 'contrastive'
                loss = self.loss_char_freqs(xenc)
                xenc2 = xenc.reshape(-1, self.num_chars)
                loss += pred_loss(xenc2, xenc2)
        return loss

def train_model(model, train_batch, steps, eval_batch, optimizer):
    model.train()
    for i in range(steps):
        optimizer.zero_grad()
        guesses = model(train_batch)
        loss = model.loss(guesses)
        loss.backward()
        optimizer.step()
        if (i % 100 == 0 or i >= steps-1) and eval_batch is not None: eval_model(model, eval_batch, loss)
    return loss

def eval_model(model, eval_batch, loss=None):
    model.eval()
    dstr1 = ''.join([CHARS[ci] for ci in torch.argmax((torch.mean(model(eval_batch), axis=0)), axis=1).cpu().numpy()])
    dstr2 = ''.join([(c1.upper() if c1 == c2 else '.') for c1, c2 in zip(dstr1, CHARS, strict=True)])
    model.train()
    if loss is None: return dstr1, dstr2
    print(f'  {dstr1} {loss.detach().cpu().numpy():0.8}\n  {dstr1}\n  {dstr2}')

################################################################ Search for America's Next Top Model
def run_test(top_model, ckpointdir, steps, test_data):
    triggers, FNORDS, NORMAL, image_data, fnord_prior, fnord = test_data
    # Test on all triggers.
    trigger_results = {}
    for trigger in triggers:
        fnords, normal = triggers[trigger]
        test_scores = fnord_test(trigger, fnords, normal, fnord_prior, top_model, CHARS)
        roc_auc, avg_prec, f1, acc, prec, rec, tpr, fpr = test_scores
        trigger_results[trigger] = (roc_auc, avg_prec, f1, acc, prec, rec, tpr, fpr)
    f1s = {k: v[2] for k, v in trigger_results.items()}
    # Get the min, max, median, mean, and std of the F1 scores.
    f1_min = min(f1s.values())
    f1_max = max(f1s.values())
    f1_median = np.median(list(f1s.values()))
    f1_mean = np.mean(list(f1s.values()))
    f1_std = np.std(list(f1s.values()))
    # Test on image classification
    image_train_acc, image_test_acc = image_test(top_model.encoder, image_data)
    # Test on the FNORDS / NORMAL
    fnord_roc_auc, fnord_avg_prec, fnord_f1, fnord_acc, fnord_prec, fnord_rec, fnord_tpr, fnord_fpr = fnord_test(fnord, FNORDS, NORMAL, fnord_prior, top_model, CHARS)

    # Append the results to f'{ckpointdir}/results.csv'
    fname = f'{ckpointdir}/results.csv'
    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['steps', 'triggers_f1_min', 'triggers_f1_max', 'triggers_f1_median', 'triggers_f1_mean', 'triggers_f1_std', 'image_train_acc', 'image_test_acc', 'fnord_roc_auc', 'fnord_avg_prec', 'fnord_f1', 'fnord_acc', 'fnord_prec', 'fnord_rec', 'fnord_tpr', 'fnord_fpr'])
    with open(fname, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([steps, f1_min, f1_max, f1_median, f1_mean, f1_std, image_train_acc, image_test_acc, fnord_roc_auc, fnord_avg_prec, fnord_f1, fnord_acc, fnord_prec, fnord_rec, fnord_tpr, fnord_fpr])

def model_search(batch, hidden, device, eval_batch, model_seeds, train_steps, bigrams, mat, char_freqs, salt, dataset, lossarity, losstype, test_data):
    ckpointdir = f'./checkpoints_{dataset}_{lossarity}_{losstype}_{salt}'
    if not os.path.exists(ckpointdir): os.makedirs(ckpointdir)
    # These are the "innate" knowledge: The bigram distribution.
    models = {} # Seed -> (model, optimizer)
    for seed in range(model_seeds):
        seed += salt * model_seeds
        torch.manual_seed(seed)
        model = WordEncoder(batch.shape[-1], hidden, bigrams, mat, char_freqs, device, lossarity, losstype).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        # Load the checkpoint if it exists.
        ckpoint = f'{ckpointdir}/model_{seed}.pt'
        optckpoint = f'{ckpointdir}/opt_{seed}.pt'
        if os.path.exists(ckpoint):
            print(f'Loading {ckpoint}')
            model.load_state_dict(torch.load(ckpoint, map_location=device))
            optimizer.load_state_dict(torch.load(optckpoint, map_location=device))
        models[seed] = (model, optimizer)
    steps_ckpt = f'{ckpointdir}/steps_{salt}.pt'
    step_start = 0
    if os.path.exists(steps_ckpt):
        step_start = torch.load(steps_ckpt)
        print(f'Loading steps from {step_start}')
    # Train every model for quick_steps
    stepsize = 100
    for steps in range(step_start, train_steps, stepsize):
        print(f'  Steps: {steps}')
        scores = {}
        for seed, (model, optimizer) in tqdm(models.items()):
            loss = train_model(model, batch, stepsize, None, optimizer)
            scores[seed] = loss.detach().cpu().numpy().item()
        printed = False
        # Show how we're doing.
        for seed, (model, _optimizer) in sorted(models.items(), key=lambda x, myscores=scores: myscores[x[0]]):
            dstr1, dstr2 = eval_model(model, eval_batch)
            if not printed:
                print(f'    {dstr1}')
                printed = True
            print(f'    {dstr2} {seed:4d} {scores[seed]:0.8f}')
        # Show test performance on test_data for top model.
        top_model, _ = models[sorted(scores.items(), key=lambda x: x[1])[0][0]]
        run_test(top_model, ckpointdir, steps+stepsize, test_data)
        # Checkpoint all the models  and the optimizers, and also record the step number to disk.
        for seed, (model, optimizer) in models.items():
            torch.save(model.state_dict(), f'{ckpointdir}/model_{seed}.pt')
            torch.save(optimizer.state_dict(), f'{ckpointdir}/opt_{seed}.pt')
        torch.save(steps+stepsize, steps_ckpt)
    # Rerun the scoring.
    scores = {}
    for seed, (model, _optimizer) in models.items():
        model.eval()
        with torch.no_grad():
            guesses = model(batch)
            loss = model.loss(guesses)
            scores[seed] = loss.detach().cpu().numpy().item()
    scores = sorted(scores.items(), key=lambda x: x[1])
    bestmodel, _ = models[scores[0][0]]
    return bestmodel, scores, models

################################################################ Main!

def single_thingle(top_model, train_batch, train_words, model_seeds, train_steps, hidden, bigrams, mat, freqs, device, eval_batch, image_data, timer, lossarity, losstype):
    ################ Cheating
    timer.lap('What happens when we cheat by using labels?')
    # Pick the best of model_seeds seeds
    best_test_loss = float('inf')
    best_cheat_model = None
    for _ in range(model_seeds):
        cheat_model = WordEncoder(eval_batch.shape[-1], hidden, bigrams, mat, freqs, device, lossarity, losstype).to(device)
        test_loss = train_model_cheat(cheat_model, image_data, train_steps*2)
        if test_loss < best_test_loss:
            print(f'New best test loss: {test_loss:0.8f}')
            image_test(cheat_model.encoder, image_data)
            best_test_loss = test_loss
            best_cheat_model = cheat_model
    print(f'Best test loss of {model_seeds} seeds: {best_test_loss:0.8f}')
    image_test(best_cheat_model.encoder, image_data)

    ################ Check the loss surface from top_model to random inits
    timer.lap('Checking loss surface...')
    top_params = list(top_model.parameters())
    model2 = WordEncoder(eval_batch.shape[-1], hidden, bigrams, mat, freqs, device, lossarity, losstype).to(device)
    model2_params = list(model2.parameters())
    dstep = .01
    for seed in range(10):
        print(f'\nSeed {seed}')
        model = WordEncoder(eval_batch.shape[-1], hidden, bigrams, mat, freqs, device, lossarity, losstype).to(device)
        model_params = list(model.parameters())
        # Linearly interpolate between model's params and top_model's params
        for blend in np.arange(0, 1+dstep, dstep):
            # blend * top_params + (1-blend) * model_params
            for mp2, mp, tp in zip(model2_params, model_params, top_params, strict=True):
                mp2.data = blend * tp.data + (1-blend) * mp.data
            # Test the model2
            model2.eval()
            loss = model2.loss(model2(train_batch)) #pylint: disable=not-callable
            print(f'  {blend:0.2f} {loss.item():0.8f}')

    ################ Clutering
    timer.lap('Clustering...')
    cluster(train_batch, train_words, CHARS)

def main(dataset, lossarity, losstype, salt):
    '''
    dataset, lossarity, losstype, salt = 'EMNIST', 'bigram', 'contrastive', 0
    dataset, lossarity, losstype, salt = 'CIFAR', 'bigram', 'contrastive', 0
    train_batch_size = 2**10  # Batch size for unsupervised training
    device = torch.device('cpu')
    char_freqs = freqs
    batch = train_batch
    '''
    ################ Constants
    # Model params: Full model is reshape -> Encoder -> logprob threshold detector
    hidden = 64               # Hidden layer size for Encoder
    # Training params
    train_batch_size = 2**20  # Batch size for unsupervised training
    train_steps = 10000       # Total training steps for each model
    model_seeds = 64         # Starting number of modelsbigrams
    # Evaluation: NOT used for optimization. Just for showing progress and for final eval.
    fnord = 'fnord'           # The only thing they fear is fnord!
    trigger_examples = 100    # Number of examples to generate for each trigger
    fnord_examples = 10000    # Number of fnord examples to generate for evaluation
    fnord_prior = .5          # Prior probability of fnord for our detection.
    eval_size = 1024          # Examples to view during training

    # Use unsupervised training to train an encoder aligned with actual labels by matching distributions.
    # Then use the encoder to detect fnords.
    print(f'Exploring {dataset} with {lossarity} and {losstype} with salt {salt}\n')

    ################ Johnny Appleseed
    device = torch.device('cuda')
    torch.manual_seed(salt); np.random.seed(salt); random.seed(salt)
    timer = LapTimer()

    ################ Load data and get batches
    timer.lap('Loading data...')
    triggers, FNORDS, NORMAL, image_data = get_all_test_data(fnord, device, dataset, trigger_examples, fnord_examples)
    test_data = triggers, FNORDS, NORMAL, image_data, fnord_prior, fnord
    # Eval data is different because we're assuming test is totally held out.
    train_batch, train_words, eval_batch, _eval_words = get_train_eval_data(device, dataset, train_batch_size, eval_size)
    bigrams, mat, freqs = get_bigram_mat()

    ################ Find the most promising encoders.
    timer.lap('Searching for modelsbigrams...')
    top_model, _scores, _modelsbigrams = model_search(train_batch, hidden, device, eval_batch, model_seeds, train_steps, bigrams, mat, freqs, salt, dataset, lossarity, losstype, test_data)

    # Just need to do this once
    if salt == 0 and lossarity == 'bigram': single_thingle(top_model, train_batch, train_words, model_seeds, train_steps, hidden, bigrams, mat, freqs, device, eval_batch, image_data, timer, lossarity, losstype)

    ################ Done!
    timer.lap('Final report.')

if __name__ == '__main__':
    # Get dataset and salt from from command line
    parser = argparse.ArgumentParser(description='Train a model to detect fnords.')
    parser.add_argument('--dataset', type=str, default='EMNIST', choices=['EMNIST', 'CIFAR'], help='The dataset to train on: EMNIST or CIFAR')
    parser.add_argument('--lossarity', type=str, default='bigram', choices=['bigram', 'unigram'] , help='Whether to use bigram or unigram features')
    parser.add_argument('--losstype', type=str, default='contrastive', choices=['contrastive', 'kldivergence'], help='The loss model: contrastive or kldivergence')
    parser.add_argument('--salt', type=int, default=0, help='The random seed salt')
    args = parser.parse_args()
    main(args.dataset, args.lossarity, args.losstype, args.salt)
