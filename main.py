"""CLI entrypoint that mirrors the official RiemannGFM `main.py` shape.

Examples
--------

Pre-train on ogbn-arxiv + Computers + Physics:

    python main.py pretrain --config configs/pretrain.yaml \
        --checkpoint checkpoints/riemann_gfm.pt

Evaluate node classification on Citeseer:

    python main.py nc --dataset citeseer --checkpoint checkpoints/riemann_gfm.pt

Evaluate link prediction on Airports:

    python main.py lp --dataset airports --checkpoint checkpoints/riemann_gfm.pt

Few-shot node classification:

    python main.py fewshot --dataset github --k 5 --checkpoint checkpoints/riemann_gfm.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml

from riemann_gfm.data.init_encoding import laplacian_positional_encoding
from riemann_gfm.data.loaders import load_dataset
from riemann_gfm.data.vocab_sampler import batch_substructures, sample_substructures
from riemann_gfm.manifolds.product_bundle import BundlePoint
from riemann_gfm.modules.model import RiemannGFM, RiemannGFMConfig
from riemann_gfm.training.fewshot import FewShotConfig, evaluate_fewshot
from riemann_gfm.training.lp import evaluate_lp
from riemann_gfm.training.nc import linear_probe
from riemann_gfm.training.pretrain import PretrainConfig, pretrain
from riemann_gfm.utils.seed import set_seed


def _model_from_config(cfg: dict) -> RiemannGFM:
    m = cfg["model"]
    return RiemannGFM(
        RiemannGFMConfig(
            d_H=m["d_H"],
            d_S=m["d_S"],
            kappa_H=m["kappa_H"],
            kappa_S=m["kappa_S"],
            n_layers=m["n_layers"],
            hidden=m["hidden"],
        )
    )


def _encode_full_graph(model: RiemannGFM, data, dim: int, device: torch.device) -> torch.Tensor:
    """Run the model over each node's substructure and return node embeddings.

    Encoding = concat(z_H, z_S) after the encoder, following Fig 5 evaluation.
    """
    feats = laplacian_positional_encoding(data.num_nodes, data.edge_index, k=dim).to(device)
    embeddings = torch.zeros(data.num_nodes, 2 * (dim + 1), device=device)
    # Iterate anchor-by-anchor to keep memory tractable on CPU dev.
    chunk = 32
    with torch.no_grad():
        for start in range(0, data.num_nodes, chunk):
            anchors = list(range(start, min(start + chunk, data.num_nodes)))
            subs = sample_substructures(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                anchors=anchors,
            )
            feats_b, tree_adj, cycle_adj, anchor_mask = batch_substructures(subs, feats)
            # Build bundle
            B, N, K = feats_b.shape
            origin_time = torch.zeros(B, N, 1, device=device)
            p_H = model.manifold_H.project(torch.cat([origin_time, feats_b], dim=-1))
            p_S = model.manifold_S.project(torch.cat([origin_time, feats_b], dim=-1))
            z_H = model.manifold_H.project_tangent(p_H, torch.cat([origin_time, feats_b], dim=-1))
            z_S = model.manifold_S.project_tangent(p_S, torch.cat([origin_time, feats_b], dim=-1))
            point = BundlePoint(p_H, z_H, p_S, z_S)
            point = model(point, tree_adj, cycle_adj)
            graph = model.graph_encoding(point, anchor_mask)
            emb = torch.cat([graph.z_H, graph.z_S], dim=-1)      # [B, 2(d+1)]
            embeddings[start : start + emb.size(0)] = emb
    return embeddings


def cmd_pretrain(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["training"]["seed"])
    model = _model_from_config(cfg)
    datasets = [load_dataset(name) for name in cfg["datasets"]]
    pc = PretrainConfig(
        dim=cfg["model"]["d_H"],
        hidden=cfg["model"]["hidden"],
        layers=cfg["model"]["n_layers"],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        batch_size=cfg["training"]["batch_size"],
        epochs=cfg["training"]["epochs"],
        iters_per_dataset=cfg["training"]["iters_per_dataset"],
        tree_depth=cfg["sampler"]["tree_depth"],
        cycle_max_len=cfg["sampler"]["cycle_max_len"],
        seed=cfg["training"]["seed"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    pretrain(model, datasets, pc)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg}, args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}")


def _load_model(checkpoint: str, device: torch.device) -> tuple[RiemannGFM, dict]:
    payload = torch.load(checkpoint, map_location=device)
    model = _model_from_config(payload["config"])
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload["config"]


def cmd_nc(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = _load_model(args.checkpoint, device)
    data = load_dataset(args.dataset).to(device)
    emb = _encode_full_graph(model, data, cfg["model"]["d_H"], device)
    report = linear_probe(emb, data.y, data.train_mask, data.test_mask)
    print(f"[NC] {args.dataset}: acc={report.accuracy:.4f} f1={report.weighted_f1:.4f}")


def cmd_lp(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = _load_model(args.checkpoint, device)
    data = load_dataset(args.dataset).to(device)
    emb = _encode_full_graph(model, data, cfg["model"]["d_H"], device)
    report = evaluate_lp(emb, data.edge_index, num_nodes=data.num_nodes)
    print(f"[LP] {args.dataset}: auc={report.auc:.4f} ap={report.ap:.4f}")


def cmd_fewshot(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = _load_model(args.checkpoint, device)
    data = load_dataset(args.dataset).to(device)
    emb = _encode_full_graph(model, data, cfg["model"]["d_H"], device)
    reports = evaluate_fewshot(
        emb, data.y, data.train_mask, data.test_mask,
        FewShotConfig(k=args.k, n_trials=args.n_trials),
    )
    accs = [r.accuracy for r in reports]
    f1s = [r.weighted_f1 for r in reports]
    print(f"[FS k={args.k}] {args.dataset}: acc={sum(accs)/len(accs):.4f} f1={sum(f1s)/len(f1s):.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="riemann-gfm")
    sub = p.add_subparsers(dest="task", required=True)

    q = sub.add_parser("pretrain")
    q.add_argument("--config", default="configs/pretrain.yaml")
    q.add_argument("--checkpoint", default="checkpoints/riemann_gfm.pt")
    q.set_defaults(func=cmd_pretrain)

    for name, fn in [("nc", cmd_nc), ("lp", cmd_lp)]:
        q = sub.add_parser(name)
        q.add_argument("--dataset", required=True)
        q.add_argument("--checkpoint", default="checkpoints/riemann_gfm.pt")
        q.set_defaults(func=fn)

    q = sub.add_parser("fewshot")
    q.add_argument("--dataset", required=True)
    q.add_argument("--k", type=int, default=5)
    q.add_argument("--n-trials", dest="n_trials", type=int, default=5)
    q.add_argument("--checkpoint", default="checkpoints/riemann_gfm.pt")
    q.set_defaults(func=cmd_fewshot)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
