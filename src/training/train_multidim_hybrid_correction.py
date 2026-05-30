#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from copy import deepcopy
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from grid_search_multidim_hybrid_correction import (
    set_seed, load_npz, build_inputs_and_baseline, HybridDataset, standardize_datasets,
    HybridCorrectionModel, hybrid_loss, run_eval
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp','lstm','gru','transformer'], required=True)
    parser.add_argument('--train_npz', required=True)
    parser.add_argument('--val_npz', required=True)
    parser.add_argument('--test_npz', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--use_log_features', action='store_true')
    parser.add_argument('--optimizer', choices=['adam','adamw'], default='adamw')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[256,256,128])
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--head_hidden', type=int, default=128)
    parser.add_argument('--head_layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=96)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--ff_dim', type=int, default=128)
    parser.add_argument('--use_cls_token', action='store_true')
    parser.add_argument('--dr_scale', type=float, default=1.0)
    parser.add_argument('--dx_scale', type=float, default=0.5)
    parser.add_argument('--lambda_sup', type=float, default=1.0)
    parser.add_argument('--lambda_res', type=float, default=0.1)
    parser.add_argument('--lambda_delta', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    train_raw = load_npz(args.train_npz)
    val_raw = load_npz(args.val_npz)
    test_raw = load_npz(args.test_npz)

    tr_seq, tr_glob, tr_y, tr_z0 = build_inputs_and_baseline(train_raw, use_log_features=args.use_log_features)
    va_seq, va_glob, va_y, va_z0 = build_inputs_and_baseline(val_raw, use_log_features=args.use_log_features)
    te_seq, te_glob, te_y, te_z0 = build_inputs_and_baseline(test_raw, use_log_features=args.use_log_features)

    train_ds = HybridDataset(tr_seq, tr_glob, tr_y, tr_z0, train_raw)
    val_ds = HybridDataset(va_seq, va_glob, va_y, va_z0, val_raw)
    test_ds = HybridDataset(te_seq, te_glob, te_y, te_z0, test_raw)
    seq_scaler, glob_scaler = standardize_datasets(train_ds, val_ds, test_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    for loader in (train_loader, val_loader, test_loader):
        loader.hp = vars(args)

    model = HybridCorrectionModel(args.model, train_ds.seq_x.shape[2], train_ds.seq_x.shape[1], train_ds.glob_x.shape[1], vars(args)).to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_rmse = float('inf'); best_epoch = -1; best_state = None; wait = 0; logs=[]
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss_sum = 0.0; train_n = 0
        for batch in train_loader:
            for k in batch: batch[k] = batch[k].to(device)
            pred, delta_vec = model(batch['seq_x'], batch['glob_x'], batch['z0'], batch['Q_total'])
            loss, _ = hybrid_loss(pred, batch['y'], delta_vec, batch, vars(args))
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            bs = pred.shape[0]; train_loss_sum += float(loss.detach().cpu().item()) * bs; train_n += bs
        val_metrics, _, _ = run_eval(model, val_loader, device)
        line = f"[Epoch {epoch:04d}] train_loss={train_loss_sum/max(train_n,1):.8f} val_loss={val_metrics['loss']:.8f} val_rmse={val_metrics['rmse']:.8f} val_mae={val_metrics['mae']:.8f} val_r2={val_metrics['r2']:.8f}"
        print(line); logs.append(line)
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']; best_epoch = epoch; best_state = deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f'[Early stopping] patience={args.patience}')
                break

    model.load_state_dict(best_state)
    val_best, _, _ = run_eval(model, val_loader, device)
    test_best, _, _ = run_eval(model, test_loader, device)
    ckpt = {
        'model_state_dict': best_state,
        'seq_scaler': seq_scaler.save(),
        'glob_scaler': glob_scaler.save(),
        'hp': vars(args),
        'seq_dim': int(train_ds.seq_x.shape[2]),
        'seq_len': int(train_ds.seq_x.shape[1]),
        'glob_dim': int(train_ds.glob_x.shape[1]),
        'best_val_rmse': float(best_val_rmse),
        'best_epoch': int(best_epoch),
    }
    torch.save(ckpt, save_dir / 'best_model.pt')
    with open(save_dir / 'metrics.txt', 'w', encoding='utf-8') as f:
        f.write('=== Best Validation / Test Summary ===\n')
        f.write(f'model: {args.model}\n')
        f.write(f'best_epoch: {best_epoch}\n')
        f.write(f'best_val_rmse: {best_val_rmse:.8f}\n\n')
        f.write('[Validation]\n')
        for k,v in val_best.items(): f.write(f'{k}: {v}\n')
        f.write('\n[Test]\n')
        for k,v in test_best.items(): f.write(f'{k}: {v}\n')
        f.write('\n[Epoch Logs]\n')
        for line in logs: f.write(line + '\n')
    with open(save_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print('\n=== Best Validation / Test Summary ===')
    print(f'model: {args.model}'); print(f'best_epoch: {best_epoch}'); print(f'best_val_rmse: {best_val_rmse:.8f}')
    print('[Validation]'); print(val_best); print('[Test]'); print(test_best)
    print(f"\n[DONE] Outputs saved to: {save_dir}")

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from copy import deepcopy
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from grid_search_multidim_hybrid_correction import (
    set_seed, load_npz, build_inputs_and_baseline, HybridDataset, standardize_datasets,
    HybridCorrectionModel, hybrid_loss, run_eval
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp','lstm','gru','transformer'], required=True)
    parser.add_argument('--train_npz', required=True)
    parser.add_argument('--val_npz', required=True)
    parser.add_argument('--test_npz', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--use_log_features', action='store_true')
    parser.add_argument('--optimizer', choices=['adam','adamw'], default='adamw')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[256,256,128])
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--head_hidden', type=int, default=128)
    parser.add_argument('--head_layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=96)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--ff_dim', type=int, default=128)
    parser.add_argument('--use_cls_token', action='store_true')
    parser.add_argument('--dr_scale', type=float, default=1.0)
    parser.add_argument('--dx_scale', type=float, default=0.5)
    parser.add_argument('--lambda_sup', type=float, default=1.0)
    parser.add_argument('--lambda_res', type=float, default=0.1)
    parser.add_argument('--lambda_delta', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    train_raw = load_npz(args.train_npz)
    val_raw = load_npz(args.val_npz)
    test_raw = load_npz(args.test_npz)

    tr_seq, tr_glob, tr_y, tr_z0 = build_inputs_and_baseline(train_raw, use_log_features=args.use_log_features)
    va_seq, va_glob, va_y, va_z0 = build_inputs_and_baseline(val_raw, use_log_features=args.use_log_features)
    te_seq, te_glob, te_y, te_z0 = build_inputs_and_baseline(test_raw, use_log_features=args.use_log_features)

    train_ds = HybridDataset(tr_seq, tr_glob, tr_y, tr_z0, train_raw)
    val_ds = HybridDataset(va_seq, va_glob, va_y, va_z0, val_raw)
    test_ds = HybridDataset(te_seq, te_glob, te_y, te_z0, test_raw)
    seq_scaler, glob_scaler = standardize_datasets(train_ds, val_ds, test_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    for loader in (train_loader, val_loader, test_loader):
        loader.hp = vars(args)

    model = HybridCorrectionModel(args.model, train_ds.seq_x.shape[2], train_ds.seq_x.shape[1], train_ds.glob_x.shape[1], vars(args)).to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_rmse = float('inf'); best_epoch = -1; best_state = None; wait = 0; logs=[]
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss_sum = 0.0; train_n = 0
        for batch in train_loader:
            for k in batch: batch[k] = batch[k].to(device)
            pred, delta_vec = model(batch['seq_x'], batch['glob_x'], batch['z0'], batch['Q_total'])
            loss, _ = hybrid_loss(pred, batch['y'], delta_vec, batch, vars(args))
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            bs = pred.shape[0]; train_loss_sum += float(loss.detach().cpu().item()) * bs; train_n += bs
        val_metrics, _, _ = run_eval(model, val_loader, device)
        line = f"[Epoch {epoch:04d}] train_loss={train_loss_sum/max(train_n,1):.8f} val_loss={val_metrics['loss']:.8f} val_rmse={val_metrics['rmse']:.8f} val_mae={val_metrics['mae']:.8f} val_r2={val_metrics['r2']:.8f}"
        print(line); logs.append(line)
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']; best_epoch = epoch; best_state = deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f'[Early stopping] patience={args.patience}')
                break

    model.load_state_dict(best_state)
    val_best, _, _ = run_eval(model, val_loader, device)
    test_best, _, _ = run_eval(model, test_loader, device)
    ckpt = {
        'model_state_dict': best_state,
        'seq_scaler': seq_scaler.save(),
        'glob_scaler': glob_scaler.save(),
        'hp': vars(args),
        'seq_dim': int(train_ds.seq_x.shape[2]),
        'seq_len': int(train_ds.seq_x.shape[1]),
        'glob_dim': int(train_ds.glob_x.shape[1]),
        'best_val_rmse': float(best_val_rmse),
        'best_epoch': int(best_epoch),
    }
    torch.save(ckpt, save_dir / 'best_model.pt')
    with open(save_dir / 'metrics.txt', 'w', encoding='utf-8') as f:
        f.write('=== Best Validation / Test Summary ===\n')
        f.write(f'model: {args.model}\n')
        f.write(f'best_epoch: {best_epoch}\n')
        f.write(f'best_val_rmse: {best_val_rmse:.8f}\n\n')
        f.write('[Validation]\n')
        for k,v in val_best.items(): f.write(f'{k}: {v}\n')
        f.write('\n[Test]\n')
        for k,v in test_best.items(): f.write(f'{k}: {v}\n')
        f.write('\n[Epoch Logs]\n')
        for line in logs: f.write(line + '\n')
    with open(save_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print('\n=== Best Validation / Test Summary ===')
    print(f'model: {args.model}'); print(f'best_epoch: {best_epoch}'); print(f'best_val_rmse: {best_val_rmse:.8f}')
    print('[Validation]'); print(val_best); print('[Test]'); print(test_best)
    print(f"\n[DONE] Outputs saved to: {save_dir}")

if __name__ == '__main__':
    main()
