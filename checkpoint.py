import torch, os

def save_checkpoint(path, model, optimizer, epoch, step, tokenizer, config, metrics):
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"step":step,"config":vars(config),"metrics":metrics,"tokenizer":tokenizer.stoi},path)
def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ck=torch.load(path,map_location=map_location); model.load_state_dict(ck["model"])
    if optimizer is not None and "optimizer" in ck: optimizer.load_state_dict(ck["optimizer"])
    return ck
