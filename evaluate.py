import argparse, math, torch
from torch.utils.data import DataLoader
from config import get_config
from tokenizer import CharTokenizer
from dataset import load_records,split_records,ConversationDataset
from model import MiniTransformer
from utils import choose_device

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",default="checkpoints/best.pt"); ap.add_argument("--data",default="data/conversations.json"); args=ap.parse_args(); device=choose_device("auto"); ck=torch.load(args.checkpoint,map_location=device); c=get_config("mini"); c.__dict__.update(ck["config"]); tok=CharTokenizer(ck["tokenizer"]); model=MiniTransformer(c).to(device); model.load_state_dict(ck["model"]); model.eval(); _,records=split_records(load_records(args.data),seed=c.seed); loader=DataLoader(ConversationDataset(records,tok,c.context_length),c.batch_size); total=n=0
 with torch.no_grad():
  for x,y in loader: _,loss=model(x.to(device),y.to(device)); total+=loss.item(); n+=1
 loss=total/max(n,1); print({"validation_loss":loss,"perplexity":math.exp(loss) if loss<20 else "overflow","note":"Generation quality requires qualitative inspection."})
if __name__=="__main__": main()
