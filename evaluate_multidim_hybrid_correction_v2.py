#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from hybrid_correction_v2_common import *
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_npz",required=True); parser.add_argument("--model",required=True); parser.add_argument("--out_dir",required=True); parser.add_argument("--tol",type=float,default=1e-12); parser.add_argument("--max_newton_iter",type=int,default=20); parser.add_argument("--device",type=str,default="cpu")
    args=parser.parse_args(); out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    data=load_npz(args.test_npz); ckpt,model,seq_scaler,glob_scaler,delta_scaler=load_model_checkpoint(args.model,device=args.device)
    hp=ckpt["hp"] if "hp" in ckpt else ckpt["args"]; use_log_features=bool(hp.get("use_log_features",False))
    seq_x,glob_x,y_true,z0,_=build_inputs_and_baseline(data,use_log_features=use_log_features); seq_shape=seq_x.shape; seq_x=apply_scaler(seq_x.reshape(-1,seq_x.shape[-1]),seq_scaler).reshape(seq_shape); glob_x=apply_scaler(glob_x,glob_scaler)
    delta_scaler_t={"mean":torch.tensor(delta_scaler["mean"].astype(np.float32),device=args.device),"std":torch.tensor(delta_scaler["std"].astype(np.float32),device=args.device)}
    preds=[]
    with torch.no_grad():
        bs=4096
        for i in range(0,len(seq_x),bs):
            s=torch.from_numpy(seq_x[i:i+bs]).to(args.device); g=torch.from_numpy(glob_x[i:i+bs]).to(args.device); z=torch.from_numpy(z0[i:i+bs].astype(np.float32)).to(args.device); qt=torch.from_numpy(np.asarray(data["Q_total"][i:i+bs]).astype(np.float32).reshape(-1,1)).to(args.device)
            pred,delta_norm,delta_real=model(s,g,z,qt,delta_scaler_t); preds.append(pred.cpu().numpy())
    pred_direct=np.concatenate(preds,axis=0); heur_pred=z0.copy(); rows=[]
    hd=vector_metrics(heur_pred,y_true); hd.update(residual_metrics(heur_pred,data)); hd["name"]="heuristic_direct"; rows.append(hd)
    href,hit,hconv=refine_batch(heur_pred.astype(np.float64),data,tol=args.tol,max_iter=args.max_newton_iter); hr=vector_metrics(href,y_true.astype(np.float64)); hr.update(residual_metrics(href,data)); hr["name"]="heuristic_plus_newton"; hr["newton_iter_mean"]=float(np.mean(hit)); hr["newton_iter_median"]=float(np.median(hit)); hr["newton_iter_p90"]=float(np.percentile(hit,90)); hr["newton_converged_ratio"]=float(np.mean(hconv)); rows.append(hr)
    nd=vector_metrics(pred_direct,y_true); nd.update(residual_metrics(pred_direct,data)); nd["name"]="neural_direct"; rows.append(nd)
    nref,nit,nconv=refine_batch(pred_direct.astype(np.float64),data,tol=args.tol,max_iter=args.max_newton_iter); nr=vector_metrics(nref,y_true.astype(np.float64)); nr.update(residual_metrics(nref,data)); nr["name"]="neural_plus_newton"; nr["newton_iter_mean"]=float(np.mean(nit)); nr["newton_iter_median"]=float(np.median(nit)); nr["newton_iter_p90"]=float(np.percentile(nit,90)); nr["newton_converged_ratio"]=float(np.mean(nconv)); rows.append(nr)
    save_csv(out_dir/"summary_metrics.csv",rows)
    per_rows=[]
    for i in range(len(y_true)):
        per_rows.append({"index":i,"true_Q1":float(y_true[i,0]),"true_x1":float(y_true[i,1]),"true_x2":float(y_true[i,2]),"heur_Q1":float(heur_pred[i,0]),"heur_x1":float(heur_pred[i,1]),"heur_x2":float(heur_pred[i,2]),"pred_Q1":float(pred_direct[i,0]),"pred_x1":float(pred_direct[i,1]),"pred_x2":float(pred_direct[i,2]),"ref_Q1":float(nref[i,0]),"ref_x1":float(nref[i,1]),"ref_x2":float(nref[i,2]),"iter":int(nit[i]),"converged":bool(nconv[i])})
    save_csv(out_dir/"per_sample_results.csv",per_rows)
    with open(out_dir/"config.json","w",encoding="utf-8") as f: json.dump({"test_npz":args.test_npz,"model_ckpt":args.model,"tol":args.tol,"max_newton_iter":args.max_newton_iter,"device":args.device,"model_name":hp["model"]},f,ensure_ascii=False,indent=2)
    print("=== Summary ==="); [print(r) for r in rows]; print(f"\\n[DONE] Outputs saved to: {out_dir.resolve()}"); print("  - summary_metrics.csv"); print("  - per_sample_results.csv"); print("  - config.json")
if __name__=="__main__": main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from hybrid_correction_v2_common import *
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_npz",required=True); parser.add_argument("--model",required=True); parser.add_argument("--out_dir",required=True); parser.add_argument("--tol",type=float,default=1e-12); parser.add_argument("--max_newton_iter",type=int,default=20); parser.add_argument("--device",type=str,default="cpu")
    args=parser.parse_args(); out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    data=load_npz(args.test_npz); ckpt,model,seq_scaler,glob_scaler,delta_scaler=load_model_checkpoint(args.model,device=args.device)
    hp=ckpt["hp"] if "hp" in ckpt else ckpt["args"]; use_log_features=bool(hp.get("use_log_features",False))
    seq_x,glob_x,y_true,z0,_=build_inputs_and_baseline(data,use_log_features=use_log_features); seq_shape=seq_x.shape; seq_x=apply_scaler(seq_x.reshape(-1,seq_x.shape[-1]),seq_scaler).reshape(seq_shape); glob_x=apply_scaler(glob_x,glob_scaler)
    delta_scaler_t={"mean":torch.tensor(delta_scaler["mean"].astype(np.float32),device=args.device),"std":torch.tensor(delta_scaler["std"].astype(np.float32),device=args.device)}
    preds=[]
    with torch.no_grad():
        bs=4096
        for i in range(0,len(seq_x),bs):
            s=torch.from_numpy(seq_x[i:i+bs]).to(args.device); g=torch.from_numpy(glob_x[i:i+bs]).to(args.device); z=torch.from_numpy(z0[i:i+bs].astype(np.float32)).to(args.device); qt=torch.from_numpy(np.asarray(data["Q_total"][i:i+bs]).astype(np.float32).reshape(-1,1)).to(args.device)
            pred,delta_norm,delta_real=model(s,g,z,qt,delta_scaler_t); preds.append(pred.cpu().numpy())
    pred_direct=np.concatenate(preds,axis=0); heur_pred=z0.copy(); rows=[]
    hd=vector_metrics(heur_pred,y_true); hd.update(residual_metrics(heur_pred,data)); hd["name"]="heuristic_direct"; rows.append(hd)
    href,hit,hconv=refine_batch(heur_pred.astype(np.float64),data,tol=args.tol,max_iter=args.max_newton_iter); hr=vector_metrics(href,y_true.astype(np.float64)); hr.update(residual_metrics(href,data)); hr["name"]="heuristic_plus_newton"; hr["newton_iter_mean"]=float(np.mean(hit)); hr["newton_iter_median"]=float(np.median(hit)); hr["newton_iter_p90"]=float(np.percentile(hit,90)); hr["newton_converged_ratio"]=float(np.mean(hconv)); rows.append(hr)
    nd=vector_metrics(pred_direct,y_true); nd.update(residual_metrics(pred_direct,data)); nd["name"]="neural_direct"; rows.append(nd)
    nref,nit,nconv=refine_batch(pred_direct.astype(np.float64),data,tol=args.tol,max_iter=args.max_newton_iter); nr=vector_metrics(nref,y_true.astype(np.float64)); nr.update(residual_metrics(nref,data)); nr["name"]="neural_plus_newton"; nr["newton_iter_mean"]=float(np.mean(nit)); nr["newton_iter_median"]=float(np.median(nit)); nr["newton_iter_p90"]=float(np.percentile(nit,90)); nr["newton_converged_ratio"]=float(np.mean(nconv)); rows.append(nr)
    save_csv(out_dir/"summary_metrics.csv",rows)
    per_rows=[]
    for i in range(len(y_true)):
        per_rows.append({"index":i,"true_Q1":float(y_true[i,0]),"true_x1":float(y_true[i,1]),"true_x2":float(y_true[i,2]),"heur_Q1":float(heur_pred[i,0]),"heur_x1":float(heur_pred[i,1]),"heur_x2":float(heur_pred[i,2]),"pred_Q1":float(pred_direct[i,0]),"pred_x1":float(pred_direct[i,1]),"pred_x2":float(pred_direct[i,2]),"ref_Q1":float(nref[i,0]),"ref_x1":float(nref[i,1]),"ref_x2":float(nref[i,2]),"iter":int(nit[i]),"converged":bool(nconv[i])})
    save_csv(out_dir/"per_sample_results.csv",per_rows)
    with open(out_dir/"config.json","w",encoding="utf-8") as f: json.dump({"test_npz":args.test_npz,"model_ckpt":args.model,"tol":args.tol,"max_newton_iter":args.max_newton_iter,"device":args.device,"model_name":hp["model"]},f,ensure_ascii=False,indent=2)
    print("=== Summary ==="); [print(r) for r in rows]; print(f"\\n[DONE] Outputs saved to: {out_dir.resolve()}"); print("  - summary_metrics.csv"); print("  - per_sample_results.csv"); print("  - config.json")
if __name__=="__main__": main()
