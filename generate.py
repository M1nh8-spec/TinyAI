import argparse, torch
from config import get_config
from tokenizer import CharTokenizer
from model import MiniTransformer
from checkpoint import load_checkpoint
from utils import choose_device

def sample(logits,temperature=0.8,top_k=20,top_p=0.9):
    logits=logits/max(temperature,1e-5)
    if top_k: values,_=torch.topk(logits,min(top_k,logits.numel())); logits[logits<values[-1]]=-float("inf")
    probs=torch.softmax(logits,dim=-1)
    if top_p<1:
        ps,ix=torch.sort(probs,descending=True); keep=torch.cumsum(ps,0)<=top_p; keep[0]=True; probs=torch.where(keep,ps,torch.zeros_like(ps)); probs=probs/probs.sum(); return ix[torch.multinomial(probs,1)]
    return torch.multinomial(probs,1)

def generate(model,tok,prompt,device,max_new_tokens=80,temperature=.8,top_k=20,top_p=.9):
    ids=torch.tensor([tok.encode(prompt,add_bos=True)],device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits,_=model(ids[:,-model.config.context_length:]); nxt=sample(logits[0,-1],temperature,top_k,top_p); ids=torch.cat([ids,nxt.view(1,1)],1)
            if nxt.item()==tok.eos_id: break
    return tok.decode(ids[0].tolist(),skip_special=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",default="checkpoints/best.pt"); ap.add_argument("--prompt"); args=ap.parse_args(); device=choose_device("auto"); ck=torch.load(args.checkpoint,map_location=device); c=get_config("mini"); c.__dict__.update(ck["config"]); tok=CharTokenizer(ck["tokenizer"]); model=MiniTransformer(c).to(device); model.load_state_dict(ck["model"])
    prompt=args.prompt
    while prompt is None or prompt.lower() not in {"quit","exit"}: prompt=prompt if prompt is not None else input("you> "); print(generate(model,tok,prompt,device)); prompt=None
if __name__=="__main__": main()
