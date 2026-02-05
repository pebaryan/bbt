import torch, math
from byte_shard_dataset import ByteShardDataset
from train_bitbyte import BitByteLM

ckpt = torch.load("ckpt.pt", map_location="cuda")
args = ckpt["args"]
seq_len = 1024  # match your training
ds = ByteShardDataset("shard_00003.bin", seq_len, seed=0, rank=0, world_size=1)
dl = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0)

model = BitByteLM(vocab_size=256, n_layer=args["n_layer"], d_model=args["d_model"],
                  n_head=args["n_head"], d_ff=args["d_ff"],
                  act_quant=not args["no_act_quant"], use_sdpa=args["use_sdpa"], ckpt=not args["no_ckpt"]).cuda()
model.load_state_dict(ckpt["model"])
model.eval()

tot_loss = 0; tot_tokens = 0
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
    for i, (x,y) in enumerate(dl):
        if i >= 200: break  # sample 200 batches; raise if you want more
        x,y = x.cuda(), y.cuda()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1,256), y.view(-1), reduction="sum")
        tot_loss += loss.item(); tot_tokens += y.numel()
bpb = (tot_loss / tot_tokens) / math.log(2)
print(f"val bpb {bpb:.4f} on {tot_tokens} tokens")