"""Shared training engine. Selection sees only the allowed validation labels."""
import csv
import ctypes
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from model import ResNet18


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    os.replace(temp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def keep_awake():
    if os.name == 'nt':
        # Prevent automatic system sleep only while this worker is alive.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)


class Data:
    def __init__(self, root, device='cuda'):
        self.root = Path(root)
        with (self.root / 'labels.csv').open(newline='', encoding='utf-8') as f:
            self.rows = list(csv.DictReader(f))
        self.train = np.loadtxt(self.root / 'train_idx.txt', dtype=np.int64)
        self.val = np.loadtxt(self.root / 'val_idx.txt', dtype=np.int64)
        assert len(self.train) == len(set(self.train)) == 45000
        assert len(self.val) == len(set(self.val)) == 5000
        assert not set(self.train) & set(self.val)
        assert set(self.train) | set(self.val) == set(range(50000))
        assert all(int(row['index']) == i for i, row in enumerate(self.rows))
        raw = np.load(self.root / 'train_images.npy')
        test = np.load(self.root / 'test_images.npy')
        assert raw.shape == (50000, 32, 32, 3) and raw.dtype == np.uint8
        assert test.shape == (10000, 32, 32, 3) and test.dtype == np.uint8
        self.device = device
        # No validation pixels or labels contribute to normalization.
        sample = raw[self.train].astype(np.float32) / 255.
        mean = sample.mean(axis=(0, 1, 2), dtype=np.float64)
        std = sample.std(axis=(0, 1, 2), dtype=np.float64)
        del sample
        self.statistics = {'mean': mean.tolist(), 'std': std.tolist()}
        self.mean = torch.tensor(mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.images = torch.from_numpy(raw).permute(0, 3, 1, 2).to(device).contiguous(memory_format=torch.channels_last)
        self.test = torch.from_numpy(test).permute(0, 3, 1, 2).to(device).contiguous(memory_format=torch.channels_last)

    def labels(self, name):
        assert name in ('noisy_label', 'synthetic_label', 'clean_label')
        return torch.tensor([int(row[name]) for row in self.rows], device=self.device, dtype=torch.long)

    def batch(self, ids, augment=False, test=False):
        x = (self.test if test else self.images)[ids].float().div_(255.)
        if augment:
            n = x.shape[0]
            x = F.pad(x, (4, 4, 4, 4), mode='reflect').permute(0, 2, 3, 1)
            dy = torch.randint(9, (n, 1, 1), device=self.device)
            dx = torch.randint(9, (n, 1, 1), device=self.device)
            rr = torch.arange(32, device=self.device).view(1, 32, 1) + dy
            cc = torch.arange(32, device=self.device).view(1, 1, 32)
            flip = torch.rand(n, 1, 1, device=self.device) < .5
            cc = torch.where(flip, 31 - cc, cc) + dx
            x = x[torch.arange(n, device=self.device).view(n, 1, 1), rr, cc].permute(0, 3, 1, 2)
        return ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)


@torch.inference_mode()
def probabilities(net, data, ids, batch_size=256, test=False):
    net.eval()
    ids = torch.as_tensor(ids, device=data.device, dtype=torch.long)
    chunks = []
    for part in ids.split(batch_size):
        with torch.autocast('cuda', dtype=torch.float16):
            logits = net(data.batch(part, test=test))
        chunks.append(logits.float().softmax(1))
    return torch.cat(chunks)


def objective(logits, y, method, ids=None, targets=None):
    logits = logits.float()
    if method == 'ce':
        return F.cross_entropy(logits, y)
    if method == 'label_smoothing':
        return F.cross_entropy(logits, y, label_smoothing=.1)
    p = logits.softmax(1).clamp(1e-6, 1 - 1e-6)
    if method == 'gce':
        q = .7
        assigned = p.gather(1, y[:, None]).squeeze(1)
        return ((1 - assigned.pow(q)) / q).mean()
    if method == 'elr':
        detached = p.detach()
        detached = detached / detached.sum(1, keepdim=True)
        targets[ids] = .7 * targets[ids] + .3 * detached
        agreement = (targets[ids] * p).sum(1).clamp(max=1 - 1e-6)
        return F.cross_entropy(logits, y) + 3. * torch.log1p(-agreement).mean()
    raise ValueError(method)


def model_state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def train(data, run_root, name, method='ce', label='noisy_label', seed=2133,
          epochs=60, batch_size=128, training_indices=None, dynamics=False,
          score_indices=None):
    """Clean labels are instantiated only for Task 1 reference or Task 2 diagnostics.

    Diagnostics cannot reach the best-checkpoint comparison: val_acc is computed
    exclusively with the allowed noisy/synthetic selection labels.
    """
    assert 1 <= epochs <= 60
    assert label != 'clean_label' or name == 'clean_reference'
    assert not dynamics or (name == 'noisy_baseline' and label == 'noisy_label')
    root = Path(run_root) / name
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'complete.json').exists():
        return json.loads((root / 'complete.json').read_text())
    keep_awake()
    seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train_ids_np = data.train if training_indices is None else np.asarray(training_indices)
    assert set(train_ids_np) <= set(data.train)
    if score_indices is not None:
        assert not set(train_ids_np) & set(score_indices)
    ids_gpu = torch.as_tensor(train_ids_np, device=data.device, dtype=torch.long)
    y = data.labels(label)
    selection_label = 'synthetic_label' if label == 'synthetic_label' else 'noisy_label'
    yval = data.labels(selection_label)[torch.as_tensor(data.val, device=data.device)]
    # These tensors are used for reporting AFTER all training updates in an epoch.
    diagnostic_clean = data.labels('clean_label') if dynamics else None
    correct_mask = diagnostic_clean[ids_gpu] == y[ids_gpu] if dynamics else None
    net = ResNet18().to(data.device, memory_format=torch.channels_last)
    opt = torch.optim.SGD(net.parameters(), lr=.1, momentum=.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')
    targets = torch.zeros(50000, 10, device=data.device) if method == 'elr' else None
    history, best_acc, best_epoch = [], -1., 0
    loss_sum = torch.zeros(len(ids_gpu), device=data.device)
    loss_count = 0
    start_epoch = 1
    checkpoint = root / 'last.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        net.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        history, best_acc, best_epoch = saved['history'], saved['best_acc'], saved['best_epoch']
        if targets is not None:
            targets.copy_(saved['targets'].to(data.device))
        loss_sum.copy_(saved['loss_sum'].to(data.device))
        loss_count = saved['loss_count']
        torch.set_rng_state(saved['rng_cpu'])
        torch.cuda.set_rng_state_all(saved['rng_cuda'])
        random.setstate(saved['rng_python'])
        np.random.set_state(saved['rng_numpy'])
        start_epoch = saved['epoch'] + 1
    settings = dict(name=name, method=method, training_label=label, selection_label=selection_label,
                    seed=seed, epochs=epochs, batch_size=batch_size, training_images=len(ids_gpu),
                    training_index_sha256=hashlib.sha256(train_ids_np.tobytes()).hexdigest(),
                    normalization=data.statistics, architecture='provided ResNet18; unchanged',
                    optimizer='SGD lr=0.1 momentum=0.9 nesterov=True weight_decay=0.0005',
                    schedule=f'cosine, T_max={epochs}, eta_min=0', augmentation='reflect-pad 4, random crop 32, horizontal flip p=0.5',
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                    precision='CUDA FP16 autocast with GradScaler; FP32 weights; channels_last',
                    cudnn_benchmark=True, bit_exact_reproducibility=False)
    atomic_json(root / 'settings.json', settings)
    np.save(root / 'training_indices.npy', train_ids_np)
    started = time.time()
    for epoch in range(start_epoch, epochs + 1):
        tick = time.time()
        net.train()
        order = ids_gpu[torch.randperm(len(ids_gpu), device=data.device)]
        running = torch.zeros((), device=data.device)
        lr = opt.param_groups[0]['lr']
        for batch_number, batch_ids in enumerate(order.split(batch_size), 1):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = net(data.batch(batch_ids, augment=True))
                loss = objective(logits, y[batch_ids], method, batch_ids, targets)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.detach() * len(batch_ids)
            if batch_number % 40 == 0:
                atomic_json(root / 'progress.json', dict(status='training', stage='training_batches',
                            epoch=epoch, batch=batch_number, batches=math.ceil(len(order)/batch_size),
                            elapsed_epoch_seconds=time.time()-tick, best_acc=best_acc, best_epoch=best_epoch))
        scheduler.step()
        atomic_json(root / 'progress.json', dict(status='training', stage='validation', epoch=epoch,
                                                best_acc=best_acc, best_epoch=best_epoch))
        valp = probabilities(net, data, data.val)
        val_acc = float((valp.argmax(1) == yval).float().mean().item() * 100)
        # Earliest epoch wins ties. Clean diagnostic metrics never select a model.
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model_state(net), root / 'best.pt')
        row = dict(epoch=epoch, loss=float(running.item() / len(ids_gpu)), lr=lr,
                   val_acc_selection=val_acc, best_acc=best_acc, best_epoch=best_epoch)
        if dynamics:
            trainp = probabilities(net, data, ids_gpu)
            pred = trainp.argmax(1)
            assigned = y[ids_gpu]
            row.update(train_acc_correct=float((pred[correct_mask] == assigned[correct_mask]).float().mean().item()*100),
                       train_acc_mislabeled=float((pred[~correct_mask] == assigned[~correct_mask]).float().mean().item()*100),
                       train_acc_mislabeled_clean=float((pred[~correct_mask] == diagnostic_clean[ids_gpu][~correct_mask]).float().mean().item()*100),
                       val_acc_noisy=val_acc,
                       val_acc_clean=float((valp.argmax(1) == diagnostic_clean[torch.as_tensor(data.val, device=data.device)]).float().mean().item()*100))
            per_loss = -trainp.gather(1, assigned[:, None]).squeeze(1).clamp_min(1e-8).log()
            if epoch <= 10:
                loss_sum += per_loss
                loss_count += 1
            np.save(root / 'final_losses.npy', per_loss.cpu().numpy())
            np.save(root / 'early_losses.npy', (loss_sum / loss_count).cpu().numpy())
        torch.cuda.synchronize()
        row['seconds'] = time.time() - tick
        history.append(row)
        atomic_json(root / 'history.json', history)
        atomic_json(root / 'progress.json', dict(status='training', **row, elapsed_seconds=time.time()-started))
        # Epoch checkpoint is replaced atomically; all optimizer/RNG/method state is retained.
        state = dict(epoch=epoch, model=model_state(net), optimizer=opt.state_dict(),
                     scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                     targets=None if targets is None else targets.cpu(), history=history,
                     best_acc=best_acc, best_epoch=best_epoch, loss_sum=loss_sum.cpu(), loss_count=loss_count,
                     rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
                     rng_python=random.getstate(), rng_numpy=np.random.get_state())
        torch.save(state, root / 'last.tmp.pt')
        os.replace(root / 'last.tmp.pt', checkpoint)
        del state
        print(f'{name}: epoch {epoch:02d}/{epochs} val={val_acc:.2f}% best={best_acc:.2f}% at {best_epoch}; {row["seconds"]:.1f}s', flush=True)
    # Keep explicit final and best snapshots; Task 1 consumes FINAL only.
    torch.save(model_state(net), root / 'final.pt')
    test_ids = np.arange(10000)
    np.save(root / 'test_final.npy', probabilities(net, data, test_ids, test=True).argmax(1).cpu().numpy())
    net.load_state_dict(torch.load(root / 'best.pt', map_location=data.device, weights_only=True))
    np.save(root / 'test_best.npy', probabilities(net, data, test_ids, test=True).argmax(1).cpu().numpy())
    if score_indices is not None:
        score_indices = np.asarray(score_indices)
        probs = probabilities(net, data, score_indices).cpu().numpy()
        noisy = np.array([int(data.rows[i]['noisy_label']) for i in score_indices])
        scores = -np.log(np.maximum(probs[np.arange(len(noisy)), noisy], 1e-8))
        np.savez(root / 'heldout_scores.npz', index=score_indices, score=scores)
    summary = dict(**settings, best_acc=best_acc, best_epoch=best_epoch,
                   final_acc=history[-1]['val_acc_selection'], elapsed_seconds=sum(r['seconds'] for r in history))
    atomic_json(root / 'complete.json', summary)
    atomic_json(root / 'progress.json', dict(status='complete', **summary))
    return summary
