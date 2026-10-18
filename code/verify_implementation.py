"""Small mathematical checks of the robust objectives; no training or GPU required."""
import torch
import torch.nn.functional as F
from engine import objective

torch.set_num_threads(2)
y = torch.tensor([0,1,2,3])
for method in ['ce','label_smoothing','gce','elr']:
    torch.manual_seed(2133)
    logits = torch.randn(4,10,requires_grad=True)
    ids = torch.tensor([3,7,11,19])
    targets = torch.zeros(20,10)
    old = targets.clone()
    value = objective(logits,y,method,ids,targets)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    if method == 'ce':
        assert torch.allclose(value,F.cross_entropy(logits,y))
    if method == 'label_smoothing':
        assert torch.allclose(value,F.cross_entropy(logits,y,label_smoothing=.1))
    if method == 'gce':
        p = logits.softmax(1)[torch.arange(4),y]
        assert torch.allclose(value,((1-p.pow(.7))/.7).mean())
    if method == 'elr':
        assert not targets.requires_grad and targets.grad_fn is None
        assert torch.allclose(targets[ids].sum(1),torch.full((4,),.3))
        other = torch.ones(20,dtype=torch.bool)
        other[ids] = False
        assert torch.equal(targets[other],old[other])
        second = torch.randn(4,10,requires_grad=True)
        objective(second,y,method,ids,targets).backward()
        assert torch.allclose(targets[ids].sum(1),torch.full((4,),.51))
print('PASS: exact CE/smoothing/GCE formulas, finite gradients, ELR detached per-image EMA and repeated backward.')
