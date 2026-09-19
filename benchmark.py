"""Small pre-training forward/backward and overfit smoke test."""
import argparse,time,torch
from config import get_config
from model import MiniTransformer

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=30); args=ap.parse_args(); c=get_config("tiny"); c.vocab_size=128; c.context_length=32; m=MiniTransformer(c); x=torch.randint(0,c.vocab_size,(2,c.context_length)); y=x.clone(); opt=torch.optim.AdamW(m.parameters(),lr=3e-3); f=b=None
 for i in range(args.steps):
  t=time.perf_counter(); _,loss=m(x,y); f=(time.perf_counter()-t) if f is None else f; t=time.perf_counter(); opt.zero_grad(); loss.backward(); opt.step(); b=(time.perf_counter()-t) if b is None else b
  if i in (0,args.steps-1): print("step",i,"loss",loss.item())
 print("parameter_report",MiniTransformer.parameter_report(m)); print("first_forward_seconds",f,"first_backward_seconds",b)
if __name__=="__main__":main()
