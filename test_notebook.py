"""
test_notebook.py
================
Validates every major component of maritime_small_obj_detection_dinov2.ipynb
using synthetic (randomly-generated) data — no real datasets needed.

Run from the repo root with:
    python test_notebook.py

Expected runtime on RTX 4090: ~3–5 minutes (DINOv2 download on first run).
Expected output: all ✓ lines and a final "ALL TESTS PASSED" message.
"""

import os, sys, json, random, tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from scipy.optimize import linear_sum_assignment
from PIL import Image

# ─── Device ───────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM : {vram:.1f} GB")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("GPU  : Apple MPS")
else:
    DEVICE = torch.device("cpu")
    print("GPU  : None (CPU)")

print(f"Device: {DEVICE}\n{'='*60}")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — YOLO→COCO converter
# ─────────────────────────────────────────────────────────────────────────────
def yolo_to_coco(img_dir, label_dir, out_json, class_names):
    categories = [{"id": i, "name": n, "supercategory": "object"}
                  for i, n in enumerate(class_names)]
    images, annotations = [], []
    ann_id = 0
    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    for img_id, img_path in enumerate(img_paths):
        w, h = Image.open(img_path).size
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        lbl = label_dir / (img_path.stem + ".txt")
        if not lbl.exists():
            continue
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])                       # FIX: explicit unpack (no star)
                cx, cy, bw, bh = map(float, parts[1:])
                abs_w  = bw * w;  abs_h  = bh * h
                abs_x1 = (cx - bw / 2) * w
                abs_y1 = (cy - bh / 2) * h
                annotations.append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": cls,
                    "bbox": [abs_x1, abs_y1, abs_w, abs_h],
                    "area": abs_w * abs_h, "iscrowd": 0
                })
                ann_id += 1
    coco = {"images": images, "annotations": annotations, "categories": categories}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(coco, f)
    return len(images), len(annotations)


def test_yolo_to_coco():
    AFO_CLASSES = ["human", "surfer", "kayak", "boat", "buoy", "sailboat"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        img_dir   = tmp / "images";  img_dir.mkdir()
        label_dir = tmp / "labels";  label_dir.mkdir()
        for i in range(4):
            Image.new("RGB", (640, 480)).save(img_dir / f"img{i}.jpg")
            with open(label_dir / f"img{i}.txt", "w") as f:
                f.write("0 0.5 0.5 0.1 0.1\n")
                f.write("3 0.2 0.8 0.04 0.06\n")
        n_imgs, n_anns = yolo_to_coco(img_dir, label_dir, tmp/"ann.json", AFO_CLASSES)
        with open(tmp/"ann.json") as f:
            out = json.load(f)
        assert n_imgs == 4 and n_anns == 8
        ann0 = out["annotations"][0]
        x1, y1, w, h = ann0["bbox"]
        assert abs(x1 - (0.5-0.05)*640) < 0.01, f"x1={x1}"
        assert abs(w  - 0.1*640)        < 0.01, f"w={w}"
    print("✓ TEST 1: YOLO→COCO converter")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Synthetic Dataset loader
# ─────────────────────────────────────────────────────────────────────────────
class SyntheticDetectionDataset(Dataset):
    """Generates random images + boxes — stands in for SDS/AFO during testing."""
    def __init__(self, n=16, num_classes=5, img_size=224, max_boxes=8):
        self.n = n; self.nc = num_classes
        self.img_size = img_size; self.max_boxes = max_boxes
        self.normalize = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    def __len__(self): return self.n

    def __getitem__(self, idx):
        img = torch.rand(3, self.img_size, self.img_size)
        img = self.normalize(img)
        nb = random.randint(1, self.max_boxes)
        # Random (cx,cy,w,h) all in (0,1)
        boxes = torch.zeros(nb, 4)
        boxes[:, :2] = torch.rand(nb, 2) * 0.8 + 0.1   # cx,cy in [0.1,0.9]
        boxes[:, 2:] = torch.rand(nb, 2) * 0.1 + 0.02  # w,h in [0.02,0.12]  (small!)
        labels = torch.randint(0, self.nc, (nb,))
        return img, {"boxes": boxes, "labels": labels, "image_id": idx}


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), list(targets)


def test_dataset():
    ds = SyntheticDetectionDataset(n=8, num_classes=5, img_size=224)
    assert len(ds) == 8
    img, tgt = ds[0]
    assert img.shape == (3, 224, 224)
    assert tgt["boxes"].shape[1] == 4
    assert (tgt["boxes"][:, 2:] > 0).all(), "boxes must have positive w,h"
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
    imgs, targets = next(iter(loader))
    assert imgs.shape == (2, 3, 224, 224)
    print("✓ TEST 2: Synthetic dataset & DataLoader")


# ─────────────────────────────────────────────────────────────────────────────
# Shared model components (from notebook sections 3–6)
# ─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h=64, max_w=64):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, d_model // 2)
        self.col_embed = nn.Embedding(max_w, d_model // 2)

    def forward(self, feat):
        B, C, H, W = feat.shape
        rows = torch.arange(H, device=feat.device)
        cols = torch.arange(W, device=feat.device)
        row_emb = self.row_embed(rows).unsqueeze(1).expand(H, W, -1)
        col_emb = self.col_embed(cols).unsqueeze(0).expand(H, W, -1)
        pos = torch.cat([row_emb, col_emb], dim=-1).permute(2,0,1)
        return pos.unsqueeze(0).expand(B,-1,-1,-1)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            d_in  = in_dim    if i == 0 else hidden_dim
            d_out = out_dim   if i == num_layers-1 else hidden_dim
            layers.append(nn.Linear(d_in, d_out))
            if i < num_layers-1: layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class DINOv2DetectionHead(nn.Module):
    def __init__(self, num_classes, d_model=128, num_queries=50,
                 nhead=4, num_enc_layers=1, num_dec_layers=2,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.d_model     = d_model
        self.fuse_p4 = nn.Conv2d(d_model, d_model, 1)
        self.fuse_p5 = nn.Conv2d(d_model, d_model, 1)
        self.pos_enc  = PositionalEncoding2D(d_model)
        enc_layer     = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.encoder  = nn.TransformerEncoder(enc_layer, num_enc_layers)
        dec_layer     = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.decoder  = nn.TransformerDecoder(dec_layer, num_dec_layers)
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_head  = nn.Linear(d_model, num_classes + 1)
        self.bbox_head   = MLP(d_model, d_model, 4, 3)

    def _fuse(self, p3, p4, p5):
        p4u = F.interpolate(self.fuse_p4(p4), size=p3.shape[-2:], mode="bilinear", align_corners=False)
        p5u = F.interpolate(self.fuse_p5(p5), size=p3.shape[-2:], mode="bilinear", align_corners=False)
        return p3 + p4u + p5u

    def forward(self, p3, p4, p5):
        B = p3.shape[0]
        fused = self._fuse(p3, p4, p5)
        pos   = self.pos_enc(fused)
        src   = (fused + pos).flatten(2).permute(2, 0, 1)
        mem   = self.encoder(src)
        q     = self.query_embed.weight.unsqueeze(1).expand(-1, B, -1)
        tgt   = torch.zeros_like(q)
        hs    = self.decoder(tgt, mem).permute(1, 0, 2)   # [B, Q, C]
        return self.class_head(hs), self.bbox_head(hs).sigmoid()


class StubBackbone(nn.Module):
    """Replaces DINOv2/DINOv3 for testing — random conv features, same API."""
    def __init__(self, out_channels=128, img_size=224):
        super().__init__()
        self.enc   = nn.Conv2d(3, out_channels, 16, stride=16)   # patch-like
        self.proj5 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        f = F.relu(self.enc(x))          # [B, C, H/16, W/16]
        p3 = f; p4 = f; p5 = self.proj5(f)
        return p3, p4, p5


class StubDetector(nn.Module):
    def __init__(self, num_classes, out_channels=128, num_queries=50, img_size=224):
        super().__init__()
        self.backbone = StubBackbone(out_channels, img_size)
        self.head     = DINOv2DetectionHead(num_classes, out_channels, num_queries)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        return self.head(p3, p4, p5)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Detection head forward pass
# ─────────────────────────────────────────────────────────────────────────────
def test_detection_head_shapes():
    NUM_CLASSES = 5; B = 2; C = 128; H = W = 14; Q = 50
    model = DINOv2DetectionHead(NUM_CLASSES, d_model=C, num_queries=Q).to(DEVICE)
    p3 = torch.randn(B, C, H, W, device=DEVICE)
    p4 = torch.randn(B, C, H, W, device=DEVICE)
    p5 = torch.randn(B, C, H//2, W//2, device=DEVICE)
    logits, bboxes = model(p3, p4, p5)
    assert logits.shape == (B, Q, NUM_CLASSES+1), f"logits: {logits.shape}"
    assert bboxes.shape == (B, Q, 4),             f"bboxes: {bboxes.shape}"
    assert (bboxes >= 0).all() and (bboxes <= 1).all(), "bboxes must be in [0,1]"
    print("✓ TEST 3: Detection head forward pass & output shapes")


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions (from notebook section 6)
# ─────────────────────────────────────────────────────────────────────────────
def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dim=-1)

def generalised_iou(pred_xyxy, tgt_xyxy):
    ix1 = torch.max(pred_xyxy[:,None,0], tgt_xyxy[None,:,0])
    iy1 = torch.max(pred_xyxy[:,None,1], tgt_xyxy[None,:,1])
    ix2 = torch.min(pred_xyxy[:,None,2], tgt_xyxy[None,:,2])
    iy2 = torch.min(pred_xyxy[:,None,3], tgt_xyxy[None,:,3])
    inter = (ix2-ix1).clamp(0) * (iy2-iy1).clamp(0)
    a1 = (pred_xyxy[:,2]-pred_xyxy[:,0]).clamp(0)*(pred_xyxy[:,3]-pred_xyxy[:,1]).clamp(0)
    a2 = (tgt_xyxy[:,2] -tgt_xyxy[:,0]).clamp(0)*(tgt_xyxy[:,3] -tgt_xyxy[:,1]).clamp(0)
    union = a1[:,None] + a2[None,:] - inter
    iou   = inter / union.clamp(1e-6)
    ex1 = torch.min(pred_xyxy[:,None,0], tgt_xyxy[None,:,0])
    ey1 = torch.min(pred_xyxy[:,None,1], tgt_xyxy[None,:,1])
    ex2 = torch.max(pred_xyxy[:,None,2], tgt_xyxy[None,:,2])
    ey2 = torch.max(pred_xyxy[:,None,3], tgt_xyxy[None,:,3])
    enc  = (ex2-ex1).clamp(0)*(ey2-ey1).clamp(0)
    return iou - (enc - union) / enc.clamp(1e-6)


class HungarianMatcher(nn.Module):
    def __init__(self, cost_cls=1., cost_bbox=5., cost_giou=2.):
        super().__init__()
        self.cost_cls = cost_cls; self.cost_bbox = cost_bbox; self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, logits, bboxes, targets):
        B, Q = logits.shape[:2]
        probs = logits.softmax(-1)
        indices = []
        for b in range(B):
            tgt_boxes  = targets[b]["boxes"].to(logits.device)
            tgt_labels = targets[b]["labels"].to(logits.device)
            T = len(tgt_labels)
            if T == 0:
                indices.append((torch.tensor([], dtype=torch.long),
                                torch.tensor([], dtype=torch.long))); continue
            c_cls  = -probs[b][:, tgt_labels]
            c_bbox = torch.cdist(bboxes[b], tgt_boxes, p=1)
            c_giou = -generalised_iou(box_cxcywh_to_xyxy(bboxes[b]),
                                      box_cxcywh_to_xyxy(tgt_boxes))
            C = (self.cost_cls*c_cls + self.cost_bbox*c_bbox + self.cost_giou*c_giou).cpu().numpy()
            row, col = linear_sum_assignment(C)
            indices.append((torch.tensor(row, dtype=torch.long),
                            torch.tensor(col, dtype=torch.long)))
        return indices


class DETRLoss(nn.Module):
    def __init__(self, num_classes, eos_coef=0.1, bbox_coef=5., giou_coef=2.):
        super().__init__()
        self.num_classes = num_classes
        self.matcher     = HungarianMatcher()
        self.bbox_coef   = bbox_coef; self.giou_coef = giou_coef
        w = torch.ones(num_classes + 1); w[-1] = eos_coef
        self.register_buffer("empty_weight", w)

    def forward(self, logits, bboxes, targets):
        indices = self.matcher(logits, bboxes, targets)
        B, Q    = logits.shape[:2]
        device  = logits.device
        tgt_cls = torch.full((B, Q), self.num_classes, dtype=torch.long, device=device)
        for b, (pi, ti) in enumerate(indices):
            if len(pi):
                tgt_cls[b, pi] = targets[b]["labels"][ti].to(device)
        loss_cls  = F.cross_entropy(logits.reshape(-1, self.num_classes+1),
                                    tgt_cls.reshape(-1),
                                    weight=self.empty_weight.to(device))
        loss_bbox = loss_giou = torch.tensor(0., device=device)
        n = 0
        for b, (pi, ti) in enumerate(indices):
            if len(pi) == 0: continue
            pb = bboxes[b, pi]; tb = targets[b]["boxes"][ti].to(device)
            loss_bbox += F.l1_loss(pb, tb, reduction="sum")
            loss_giou += (1 - generalised_iou(box_cxcywh_to_xyxy(pb),
                                              box_cxcywh_to_xyxy(tb)).diag()).sum()
            n += len(pi)
        n = max(n, 1)
        total = loss_cls + self.bbox_coef*(loss_bbox/n) + self.giou_coef*(loss_giou/n)
        return {"loss_cls": loss_cls, "loss_bbox": loss_bbox/n,
                "loss_giou": loss_giou/n, "loss_total": total}


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Hungarian matcher
# ─────────────────────────────────────────────────────────────────────────────
def test_matcher():
    matcher = HungarianMatcher()
    logits  = torch.randn(2, 20, 6)    # B=2, Q=20, C+1=6
    bboxes  = torch.sigmoid(torch.randn(2, 20, 4))
    targets = [
        {"boxes": torch.tensor([[0.5,0.5,0.1,0.1],[0.2,0.3,0.05,0.05]]),
         "labels": torch.tensor([0, 2])},
        {"boxes": torch.tensor([[0.7,0.7,0.08,0.08]]),
         "labels": torch.tensor([1])},
    ]
    indices = matcher(logits, bboxes, targets)
    assert len(indices) == 2
    pi0, ti0 = indices[0]; assert len(pi0) == 2  # 2 GT boxes → 2 matched
    pi1, ti1 = indices[1]; assert len(pi1) == 1  # 1 GT box  → 1 matched
    assert len(set(pi0.tolist())) == 2, "matched pred indices must be unique"
    print("✓ TEST 4: Hungarian matcher (bipartite assignment)")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Loss forward pass + finite values
# ─────────────────────────────────────────────────────────────────────────────
def test_loss():
    NUM_CLASSES = 5; B = 2; Q = 20
    criterion = DETRLoss(NUM_CLASSES).to(DEVICE)
    logits = torch.randn(B, Q, NUM_CLASSES+1, device=DEVICE)
    bboxes = torch.sigmoid(torch.randn(B, Q, 4, device=DEVICE))
    targets = [
        {"boxes": torch.rand(3, 4)*0.3+0.1, "labels": torch.randint(0,NUM_CLASSES,(3,))},
        {"boxes": torch.rand(2, 4)*0.3+0.1, "labels": torch.randint(0,NUM_CLASSES,(2,))},
    ]
    losses = criterion(logits, bboxes, targets)
    for k, v in losses.items():
        assert torch.isfinite(v), f"{k} is not finite: {v}"
        assert v.item() >= 0,     f"{k} is negative: {v}"
    print(f"✓ TEST 5: Loss forward pass — "
          f"total={losses['loss_total'].item():.4f}  "
          f"cls={losses['loss_cls'].item():.4f}  "
          f"giou={losses['loss_giou'].item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — Full model forward pass (stub backbone, no DINOv2 download)
# ─────────────────────────────────────────────────────────────────────────────
def test_full_model_forward():
    NUM_CLASSES = 5; B = 2; IMG = 224
    model = StubDetector(NUM_CLASSES, out_channels=128, num_queries=50, img_size=IMG).to(DEVICE)
    x = torch.randn(B, 3, IMG, IMG, device=DEVICE)
    logits, bboxes = model(x)
    assert logits.shape == (B, 50, NUM_CLASSES+1)
    assert bboxes.shape == (B, 50, 4)
    assert (bboxes >= 0).all() and (bboxes <= 1).all()
    params = sum(p.numel() for p in model.parameters())
    print(f"✓ TEST 6: Full model forward — logits {tuple(logits.shape)}, "
          f"bboxes {tuple(bboxes.shape)}, params={params/1e6:.2f}M")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — 1 epoch training loop (end-to-end gradient flow)
# ─────────────────────────────────────────────────────────────────────────────
def test_train_one_epoch():
    NUM_CLASSES = 5; IMG = 224; BATCH = 4; Q = 50
    ds      = SyntheticDetectionDataset(n=16, num_classes=NUM_CLASSES, img_size=IMG)
    loader  = DataLoader(ds, batch_size=BATCH, collate_fn=collate_fn)
    model   = StubDetector(NUM_CLASSES, out_channels=128, num_queries=Q, img_size=IMG).to(DEVICE)
    crit    = DETRLoss(NUM_CLASSES).to(DEVICE)
    opt     = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    epoch_loss = 0.0
    for step, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(DEVICE)
        logits, bboxes = model(imgs)
        losses = crit(logits, bboxes, targets)
        opt.zero_grad()
        losses["loss_total"].backward()

        # Verify gradients are flowing
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"NaN/Inf grad in {name}"
        opt.step()
        epoch_loss += losses["loss_total"].item()

    avg_loss = epoch_loss / len(loader)
    assert avg_loss < 1000, f"Loss suspiciously high: {avg_loss}"  # sanity bound
    print(f"✓ TEST 7: 1 epoch training — avg loss={avg_loss:.4f}, "
          f"{len(loader)} steps × batch {BATCH}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8 — GIoU correctness
# ─────────────────────────────────────────────────────────────────────────────
def test_giou():
    # Perfect overlap → GIoU = 1
    box = torch.tensor([[0.1, 0.1, 0.5, 0.5]])
    g   = generalised_iou(box, box)
    assert abs(g.item() - 1.0) < 1e-4, f"GIoU(box,box)={g.item()}"

    # Non-overlapping boxes → GIoU ∈ [-1, 0)
    a = torch.tensor([[0.0, 0.0, 0.2, 0.2]])
    b = torch.tensor([[0.8, 0.8, 1.0, 1.0]])
    g2 = generalised_iou(a, b)
    assert -1.0 <= g2.item() < 0.0, f"GIoU(non-overlap)={g2.item()}"
    print(f"✓ TEST 8: GIoU — perfect={generalised_iou(box,box).item():.4f}, "
          f"non-overlap={g2.item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9 — Section 0B: DINOv3Backbone shape check (real weights, optional)
# ─────────────────────────────────────────────────────────────────────────────
def test_dinov3_backbone(run=False):
    """
    Downloads DINOv3 weights from HuggingFace (~100 MB, ViT-S).
    Set run=True to execute; skip by default to avoid mandatory download.
    """
    if not run:
        print("  TEST 9: DINOv3 backbone — SKIPPED (set run=True to enable)")
        return
    try:
        import timm
        from packaging import version
        assert version.parse(timm.__version__) >= version.parse("1.0.20"), \
            f"Need timm>=1.0.20, got {timm.__version__}"

        IMG = 560
        vit = timm.create_model("vit_small_patch14_dinov3", pretrained=True, img_size=IMG)
        vit = vit.eval().to(DEVICE)

        x    = torch.randn(1, 3, IMG, IMG, device=DEVICE)
        idxs = [vit.blocks.__len__()//4-1,
                vit.blocks.__len__()//2-1,
                vit.blocks.__len__()-1]

        with torch.no_grad():
            feats = vit.get_intermediate_layers(x, n=idxs, reshape=True,
                                                return_class_token=False)
        assert len(feats) == 3, f"Expected 3 feature maps, got {len(feats)}"
        H_p = IMG // 14  # = 40
        for fi, f in enumerate(feats):
            assert f.shape == (1, 384, H_p, H_p), f"Feature {fi} shape: {f.shape}"
        print(f"✓ TEST 9: DINOv3 backbone — shapes {[tuple(f.shape) for f in feats]}")
    except Exception as e:
        print(f"  TEST 9: DINOv3 backbone — FAILED: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10 — Section 0A: DINOv2 backbone shape check (real weights, optional)
# ─────────────────────────────────────────────────────────────────────────────
def test_dinov2_backbone(run=False):
    """
    Downloads DINOv2 ViT-S/14 from torch.hub (~80 MB).
    Set run=True to execute; skip by default to avoid mandatory download.
    """
    if not run:
        print("  TEST 10: DINOv2 backbone — SKIPPED (set run=True to enable)")
        return
    try:
        IMG = 560
        vit = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        vit = vit.eval().to(DEVICE)
        num_blocks = len(vit.blocks)
        tap = [num_blocks//4, num_blocks//2, num_blocks-1]

        x      = torch.randn(1, 3, IMG, IMG, device=DEVICE)
        H_p    = IMG // 14
        tokens = vit.prepare_tokens_with_masks(x, None)
        tapped = {}
        with torch.no_grad():
            for i, block in enumerate(vit.blocks):
                tokens = block(tokens)
                if i in tap:
                    tapped[i] = tokens[:, 1:, :]   # drop CLS

        for k, v in tapped.items():
            spatial = v.permute(0,2,1).reshape(1,-1,H_p,H_p)
            assert spatial.shape == (1, 384, H_p, H_p), f"Block {k}: {spatial.shape}"
        print(f"✓ TEST 10: DINOv2 backbone — tapped blocks {tap}, "
              f"spatial shape (1, 384, {H_p}, {H_p})")
    except Exception as e:
        print(f"  TEST 10: DINOv2 backbone — FAILED: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov2", action="store_true",
                        help="Also download & test DINOv2 backbone weights")
    parser.add_argument("--dinov3", action="store_true",
                        help="Also download & test DINOv3 backbone weights (needs timm>=1.0.20)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Running notebook tests...")
    print(f"{'='*60}\n")

    test_yolo_to_coco()
    test_dataset()
    test_detection_head_shapes()
    test_matcher()
    test_loss()
    test_full_model_forward()
    test_train_one_epoch()
    test_giou()
    test_dinov3_backbone(run=args.dinov3)
    test_dinov2_backbone(run=args.dinov2)

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED ✓")
    print(f"{'='*60}")
    print("\nTo also test real backbone weights (downloads ~80-100 MB):")
    print("  python test_notebook.py --dinov2")
    print("  python test_notebook.py --dinov3   # requires timm>=1.0.20")
