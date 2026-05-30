#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from hybrid_correction_v2_common import *
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train_npz",required=True); parser.add_argument("--val_npz",required=True); parser.add_argument("--test_npz",required=True); parser.add_argument("--out_dir",required=True)
    parser.add_argument("--models",nargs="+",default=["mlp","lstm","gru","transformer"]); parser.add_argument("--epochs",type=int,default=120); parser.add_argument("--batch_size",type=int,default=256); parser.add_argument("--patience",type=int,default=20); parser.add_argument("--seed",type=int,default=42); parser.add_argument("--device",type=str,default="cpu"); parser.add_argument("--num_workers",type=int,default=0)
    parser.add_argument("--tol",type=float,default=1e-12); parser.add_argument("--max_newton_iter",type=int,default=20)
    parser.add_argument("--rank_metric",default="plus_newton_r2",choices=["direct_r2","direct_rmse","direct_mae","plus_newton_r2","plus_newton_rmse","plus_newton_mae","plus_newton_converged_ratio"])
    args=parser.parse_args()
    set_seed(args.seed); device=torch.device(args.device); out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    train_raw=load_npz(args.train_npz); val_raw=load_npz(args.val_npz); test_raw=load_npz(args.test_npz)
    search_space=build_search_space(args.models); all_rows=[]; best_metric=None; best_row=None; best_ckpt=None
    for trial_id,hp in enumerate(search_space,start=1):
        trial_name=f"trial_{trial_id:03d}_{hp['model']}"; print(f"\\n========== {trial_name} =========="); print(json.dumps(hp,ensure_ascii=False))
        tr_seq,tr_glob,tr_y,tr_z0,tr_delta=build_inputs_and_baseline(train_raw,use_log_features=hp["use_log_features"])
        va_seq,va_glob,va_y,va_z0,va_delta=build_inputs_and_baseline(val_raw,use_log_features=hp["use_log_features"])
        te_seq,te_glob,te_y,te_z0,te_delta=build_inputs_and_baseline(test_raw,use_log_features=hp["use_log_features"])
        train_ds=HybridDataset(tr_seq,tr_glob,tr_y,tr_z0,tr_delta,train_raw); val_ds=HybridDataset(va_seq,va_glob,va_y,va_z0,va_delta,val_raw); test_ds=HybridDataset(te_seq,te_glob,te_y,te_z0,te_delta,test_raw)
        seq_scaler,glob_scaler,delta_scaler=standardize_datasets(train_ds,val_ds,test_ds)
        delta_scaler_t={"mean":torch.tensor(delta_scaler.mean.astype(np.float32),device=device),"std":torch.tensor(delta_scaler.std.astype(np.float32),device=device)}
        train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers); val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers); test_loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
        model=HybridCorrectionModel(hp["model"],train_ds.seq_x.shape[2],train_ds.seq_x.shape[1],train_ds.glob_x.shape[1],hp).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=hp["lr"],weight_decay=hp["weight_decay"]) if hp["optimizer"]=="adamw" else torch.optim.Adam(model.parameters(),lr=hp["lr"],weight_decay=hp["weight_decay"])
        best_val_rmse=float("inf"); best_epoch=-1; best_state=None; wait=0; start=time.time()
        for epoch in range(1,args.epochs+1):
            model.train(); train_loss_sum=0.0; train_n=0
            for batch in train_loader:
                for k in batch: batch[k]=batch[k].to(device)
                pred,delta_norm,delta_real=model(batch["seq_x"],batch["glob_x"],batch["z0"],batch["Q_total"],delta_scaler_t)
                loss=delta_supervised_loss(delta_norm,batch["delta_target"],loss_name=hp["loss_name"])
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
                bs=pred.shape[0]; train_loss_sum += float(loss.detach().cpu().item())*bs; train_n += bs
            val_metrics,_,_=run_eval(model,val_loader,hp["loss_name"],device,delta_scaler_t)
            print(f"[{trial_name}] epoch={epoch:03d} train_loss={train_loss_sum/max(train_n,1):.6f} val_rmse={val_metrics['rmse']:.6f} val_r2={val_metrics['r2']:.6f}")
            if val_metrics["rmse"]<best_val_rmse:
                best_val_rmse=val_metrics["rmse"]; best_epoch=epoch; best_state=deepcopy(model.state_dict()); wait=0
            else:
                wait += 1
                if wait>=args.patience: break
        model.load_state_dict(best_state)
        direct_metrics,pred_direct,true=run_eval(model,test_loader,hp["loss_name"],device,delta_scaler_t); direct_metrics.update(residual_metrics(pred_direct,test_raw))
        pred_ref,pred_iter,pred_conv=refine_batch(pred_direct.astype(np.float64),test_raw,tol=args.tol,max_iter=args.max_newton_iter)
        plus=vector_metrics(pred_ref,true.astype(np.float64)); plus.update(residual_metrics(pred_ref,test_raw)); plus["newton_iter_mean"]=float(np.mean(pred_iter)); plus["newton_iter_median"]=float(np.median(pred_iter)); plus["newton_iter_p90"]=float(np.percentile(pred_iter,90)); plus["newton_converged_ratio"]=float(np.mean(pred_conv))
        row={"trial_id":trial_id,"trial_name":trial_name,"model":hp["model"],"best_epoch":best_epoch,"elapsed_sec":time.time()-start,"direct_mae":direct_metrics["mae"],"direct_rmse":direct_metrics["rmse"],"direct_r2":direct_metrics["r2"],"direct_valid_ratio":direct_metrics["valid_ratio"],"direct_residual_mean":direct_metrics["residual_mean"],"direct_residual_median":direct_metrics["residual_median"],"direct_residual_p90":direct_metrics["residual_p90"],"plus_newton_mae":plus["mae"],"plus_newton_rmse":plus["rmse"],"plus_newton_r2":plus["r2"],"plus_newton_valid_ratio":plus["valid_ratio"],"plus_newton_residual_mean":plus["residual_mean"],"plus_newton_residual_median":plus["residual_median"],"plus_newton_residual_p90":plus["residual_p90"],"plus_newton_newton_iter_mean":plus["newton_iter_mean"],"plus_newton_newton_iter_median":plus["newton_iter_median"],"plus_newton_newton_iter_p90":plus["newton_iter_p90"],"plus_newton_converged_ratio":plus["newton_converged_ratio"],"hp_use_log_features":hp["use_log_features"],"hp_optimizer":hp["optimizer"],"hp_loss_name":hp["loss_name"],"hp_dropout":hp["dropout"],"hp_lr":hp["lr"],"hp_weight_decay":hp["weight_decay"],"hp_hidden_dims":json.dumps(hp["hidden_dims"]),"hp_hidden_size":hp["hidden_size"],"hp_num_layers":hp["num_layers"],"hp_head_hidden":hp["head_hidden"],"hp_head_layers":hp["head_layers"],"hp_d_model":hp["d_model"],"hp_nhead":hp["nhead"],"hp_ff_dim":hp["ff_dim"],"hp_use_cls_token":hp["use_cls_token"]}
        all_rows.append(row)
        cur_metric=row[args.rank_metric]
        better = best_metric is None or ((cur_metric<best_metric) if args.rank_metric in ["direct_rmse","direct_mae","plus_newton_rmse","plus_newton_mae"] else (cur_metric>best_metric))
        if better:
            best_metric=cur_metric; best_row=dict(row); best_ckpt={"model_state_dict":deepcopy(model.state_dict()),"seq_scaler":seq_scaler.save(),"glob_scaler":glob_scaler.save(),"delta_scaler":delta_scaler.save(),"hp":hp,"seq_dim":train_ds.seq_x.shape[2],"seq_len":train_ds.seq_x.shape[1],"glob_dim":train_ds.glob_x.shape[1],"best_val_rmse":best_val_rmse,"best_epoch":best_epoch}
        with open(out_dir/f"{trial_name}.json","w",encoding="utf-8") as f: json.dump(row,f,ensure_ascii=False,indent=2)
    if not all_rows: raise RuntimeError("No successful trials completed.")
    reverse=args.rank_metric not in ["direct_rmse","direct_mae","plus_newton_rmse","plus_newton_mae"]; all_rows_sorted=sorted(all_rows,key=lambda r:r[args.rank_metric],reverse=reverse)
    save_csv(out_dir/"all_trials.csv",all_rows_sorted)
    with open(out_dir/"best_result.json","w",encoding="utf-8") as f: json.dump(best_row,f,ensure_ascii=False,indent=2)
    torch.save(best_ckpt,out_dir/"best_model_by_grid.pt")
    print("\\n================ FINAL RANKING ================")
    for row in all_rows_sorted[:10]:
        print({"trial_id":row["trial_id"],"model":row["model"],args.rank_metric:row[args.rank_metric],"plus_newton_rmse":row["plus_newton_rmse"],"plus_newton_r2":row["plus_newton_r2"],"plus_newton_converged_ratio":row["plus_newton_converged_ratio"]})
    print("\\n[DONE]"); print(out_dir/"all_trials.csv"); print(out_dir/"best_result.json"); print(out_dir/"best_model_by_grid.pt")
if __name__=="__main__": main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from hybrid_correction_v2_common import *
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train_npz",required=True); parser.add_argument("--val_npz",required=True); parser.add_argument("--test_npz",required=True); parser.add_argument("--out_dir",required=True)
    parser.add_argument("--models",nargs="+",default=["mlp","lstm","gru","transformer"]); parser.add_argument("--epochs",type=int,default=120); parser.add_argument("--batch_size",type=int,default=256); parser.add_argument("--patience",type=int,default=20); parser.add_argument("--seed",type=int,default=42); parser.add_argument("--device",type=str,default="cpu"); parser.add_argument("--num_workers",type=int,default=0)
    parser.add_argument("--tol",type=float,default=1e-12); parser.add_argument("--max_newton_iter",type=int,default=20)
    parser.add_argument("--rank_metric",default="plus_newton_r2",choices=["direct_r2","direct_rmse","direct_mae","plus_newton_r2","plus_newton_rmse","plus_newton_mae","plus_newton_converged_ratio"])
    args=parser.parse_args()
    set_seed(args.seed); device=torch.device(args.device); out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    train_raw=load_npz(args.train_npz); val_raw=load_npz(args.val_npz); test_raw=load_npz(args.test_npz)
    search_space=build_search_space(args.models); all_rows=[]; best_metric=None; best_row=None; best_ckpt=None
    for trial_id,hp in enumerate(search_space,start=1):
        trial_name=f"trial_{trial_id:03d}_{hp['model']}"; print(f"\\n========== {trial_name} =========="); print(json.dumps(hp,ensure_ascii=False))
        tr_seq,tr_glob,tr_y,tr_z0,tr_delta=build_inputs_and_baseline(train_raw,use_log_features=hp["use_log_features"])
        va_seq,va_glob,va_y,va_z0,va_delta=build_inputs_and_baseline(val_raw,use_log_features=hp["use_log_features"])
        te_seq,te_glob,te_y,te_z0,te_delta=build_inputs_and_baseline(test_raw,use_log_features=hp["use_log_features"])
        train_ds=HybridDataset(tr_seq,tr_glob,tr_y,tr_z0,tr_delta,train_raw); val_ds=HybridDataset(va_seq,va_glob,va_y,va_z0,va_delta,val_raw); test_ds=HybridDataset(te_seq,te_glob,te_y,te_z0,te_delta,test_raw)
        seq_scaler,glob_scaler,delta_scaler=standardize_datasets(train_ds,val_ds,test_ds)
        delta_scaler_t={"mean":torch.tensor(delta_scaler.mean.astype(np.float32),device=device),"std":torch.tensor(delta_scaler.std.astype(np.float32),device=device)}
        train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers); val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers); test_loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
        model=HybridCorrectionModel(hp["model"],train_ds.seq_x.shape[2],train_ds.seq_x.shape[1],train_ds.glob_x.shape[1],hp).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=hp["lr"],weight_decay=hp["weight_decay"]) if hp["optimizer"]=="adamw" else torch.optim.Adam(model.parameters(),lr=hp["lr"],weight_decay=hp["weight_decay"])
        best_val_rmse=float("inf"); best_epoch=-1; best_state=None; wait=0; start=time.time()
        for epoch in range(1,args.epochs+1):
            model.train(); train_loss_sum=0.0; train_n=0
            for batch in train_loader:
                for k in batch: batch[k]=batch[k].to(device)
                pred,delta_norm,delta_real=model(batch["seq_x"],batch["glob_x"],batch["z0"],batch["Q_total"],delta_scaler_t)
                loss=delta_supervised_loss(delta_norm,batch["delta_target"],loss_name=hp["loss_name"])
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
                bs=pred.shape[0]; train_loss_sum += float(loss.detach().cpu().item())*bs; train_n += bs
            val_metrics,_,_=run_eval(model,val_loader,hp["loss_name"],device,delta_scaler_t)
            print(f"[{trial_name}] epoch={epoch:03d} train_loss={train_loss_sum/max(train_n,1):.6f} val_rmse={val_metrics['rmse']:.6f} val_r2={val_metrics['r2']:.6f}")
            if val_metrics["rmse"]<best_val_rmse:
                best_val_rmse=val_metrics["rmse"]; best_epoch=epoch; best_state=deepcopy(model.state_dict()); wait=0
            else:
                wait += 1
                if wait>=args.patience: break
        model.load_state_dict(best_state)
        direct_metrics,pred_direct,true=run_eval(model,test_loader,hp["loss_name"],device,delta_scaler_t); direct_metrics.update(residual_metrics(pred_direct,test_raw))
        pred_ref,pred_iter,pred_conv=refine_batch(pred_direct.astype(np.float64),test_raw,tol=args.tol,max_iter=args.max_newton_iter)
        plus=vector_metrics(pred_ref,true.astype(np.float64)); plus.update(residual_metrics(pred_ref,test_raw)); plus["newton_iter_mean"]=float(np.mean(pred_iter)); plus["newton_iter_median"]=float(np.median(pred_iter)); plus["newton_iter_p90"]=float(np.percentile(pred_iter,90)); plus["newton_converged_ratio"]=float(np.mean(pred_conv))
        row={"trial_id":trial_id,"trial_name":trial_name,"model":hp["model"],"best_epoch":best_epoch,"elapsed_sec":time.time()-start,"direct_mae":direct_metrics["mae"],"direct_rmse":direct_metrics["rmse"],"direct_r2":direct_metrics["r2"],"direct_valid_ratio":direct_metrics["valid_ratio"],"direct_residual_mean":direct_metrics["residual_mean"],"direct_residual_median":direct_metrics["residual_median"],"direct_residual_p90":direct_metrics["residual_p90"],"plus_newton_mae":plus["mae"],"plus_newton_rmse":plus["rmse"],"plus_newton_r2":plus["r2"],"plus_newton_valid_ratio":plus["valid_ratio"],"plus_newton_residual_mean":plus["residual_mean"],"plus_newton_residual_median":plus["residual_median"],"plus_newton_residual_p90":plus["residual_p90"],"plus_newton_newton_iter_mean":plus["newton_iter_mean"],"plus_newton_newton_iter_median":plus["newton_iter_median"],"plus_newton_newton_iter_p90":plus["newton_iter_p90"],"plus_newton_converged_ratio":plus["newton_converged_ratio"],"hp_use_log_features":hp["use_log_features"],"hp_optimizer":hp["optimizer"],"hp_loss_name":hp["loss_name"],"hp_dropout":hp["dropout"],"hp_lr":hp["lr"],"hp_weight_decay":hp["weight_decay"],"hp_hidden_dims":json.dumps(hp["hidden_dims"]),"hp_hidden_size":hp["hidden_size"],"hp_num_layers":hp["num_layers"],"hp_head_hidden":hp["head_hidden"],"hp_head_layers":hp["head_layers"],"hp_d_model":hp["d_model"],"hp_nhead":hp["nhead"],"hp_ff_dim":hp["ff_dim"],"hp_use_cls_token":hp["use_cls_token"]}
        all_rows.append(row)
        cur_metric=row[args.rank_metric]
        better = best_metric is None or ((cur_metric<best_metric) if args.rank_metric in ["direct_rmse","direct_mae","plus_newton_rmse","plus_newton_mae"] else (cur_metric>best_metric))
        if better:
            best_metric=cur_metric; best_row=dict(row); best_ckpt={"model_state_dict":deepcopy(model.state_dict()),"seq_scaler":seq_scaler.save(),"glob_scaler":glob_scaler.save(),"delta_scaler":delta_scaler.save(),"hp":hp,"seq_dim":train_ds.seq_x.shape[2],"seq_len":train_ds.seq_x.shape[1],"glob_dim":train_ds.glob_x.shape[1],"best_val_rmse":best_val_rmse,"best_epoch":best_epoch}
        with open(out_dir/f"{trial_name}.json","w",encoding="utf-8") as f: json.dump(row,f,ensure_ascii=False,indent=2)
    if not all_rows: raise RuntimeError("No successful trials completed.")
    reverse=args.rank_metric not in ["direct_rmse","direct_mae","plus_newton_rmse","plus_newton_mae"]; all_rows_sorted=sorted(all_rows,key=lambda r:r[args.rank_metric],reverse=reverse)
    save_csv(out_dir/"all_trials.csv",all_rows_sorted)
    with open(out_dir/"best_result.json","w",encoding="utf-8") as f: json.dump(best_row,f,ensure_ascii=False,indent=2)
    torch.save(best_ckpt,out_dir/"best_model_by_grid.pt")
    print("\\n================ FINAL RANKING ================")
    for row in all_rows_sorted[:10]:
        print({"trial_id":row["trial_id"],"model":row["model"],args.rank_metric:row[args.rank_metric],"plus_newton_rmse":row["plus_newton_rmse"],"plus_newton_r2":row["plus_newton_r2"],"plus_newton_converged_ratio":row["plus_newton_converged_ratio"]})
    print("\\n[DONE]"); print(out_dir/"all_trials.csv"); print(out_dir/"best_result.json"); print(out_dir/"best_model_by_grid.pt")
if __name__=="__main__": main()
