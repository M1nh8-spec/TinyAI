"""Train from scratch. No pretrained artifacts are loaded."""
import argparse, os, torch
from torch.utils.data import DataLoader
from config import get_config
from tokenizer import CharTokenizer
from dataset import load_records,split_records,ConversationDataset
from model import MiniTransformer
from checkpoint import save_checkpoint,load_checkpoint
from utils import seed_all,choose_device,set_cpu_threads

def run_epoch(model,loader,opt,device,train=True):
    model.train(train); total=0.; count=0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        with torch.set_grad_enabled(train):
            _,loss=model(x,y)
            if train: opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        total+=loss.item(); count+=1
    return total/max(count,1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="mini"); ap.add_argument("--data",default="data/conversations.json"); ap.add_argument("--epochs",type=int); ap.add_argument("--resume"); ap.add_argument("--out",default="checkpoints"); ap.add_argument("--device",default=None); args=ap.parse_args()
    c=get_config(args.config); c.epochs=args.epochs or c.epochs; seed_all(c.seed); set_cpu_threads(c.num_threads); device=choose_device(args.device or c.device)
    records=load_records(args.data); tok=CharTokenizer().fit([t["text"] for r in records for t in r["conversation"]]+["<BOS> <EOS> <USER> <ASSISTANT> <VI> <EN>"]); c.vocab_size=len(tok.stoi)
    tr,va=split_records(records,seed=c.seed); train=DataLoader(ConversationDataset(tr,tok,c.context_length),c.batch_size,shuffle=True,num_workers=c.num_workers); valid=DataLoader(ConversationDataset(va,tok,c.context_length),c.batch_size,num_workers=c.num_workers)
    model=MiniTransformer(c).to(device); opt=torch.optim.AdamW(model.parameters(),lr=c.learning_rate); start=0
    if args.resume: start=load_checkpoint(args.resume,model,opt,device).get("epoch",0)+1
    print("device",device,"parameters",MiniTransformer.parameter_report(model)); os.makedirs(args.out,exist_ok=True); best=float("inf")
    for epoch in range(start,c.epochs):
        tl=run_epoch(model,train,opt,device,True); vl=run_epoch(model,valid,opt,device,False); print(f"epoch={epoch} train_loss={tl:.4f} val_loss={vl:.4f} lr={c.learning_rate}")
        save_checkpoint(os.path.join(args.out,"last.pt"),model,opt,epoch,epoch*len(train),tok,c,{"train_loss":tl,"val_loss":vl})
        if vl<best: best=vl; save_checkpoint(os.path.join(args.out,"best.pt"),model,opt,epoch,epoch*len(train),tok,c,{"train_loss":tl,"val_loss":vl})
if __name__=="__main__": main()
