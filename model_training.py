from tqdm import tqdm
import d3rlpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from model import VAE,AutoEncoder
import wandb
import argparse

parser = argparse.ArgumentParser(description='VAE Trajectory Data')
parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--model_name", type=str, default="hopper_ae", help="Name of the model")
parser.add_argument("--dataset", type=str, default="hopper-medium-replay-v0", help="Name of the dataset")
parser.add_argument("--model_type", type=str, default="autoencoder", help="Type of the model")
parser.add_argument("--train_from_scratch", action="store_true", help="Train the model from scratch")
args = parser.parse_args()
if args.use_wandb:
    wandb.init(project="vae-trajectory")
else:
    print("Wandb initialization skipped due to --use_wandb False.")
if args.device > 0:
    device = torch.device(f"cuda:{args.device}")

# 参数设定
input_dim = 24  # 每个时间步的向量维度
hidden_dim = 128  # 隐藏层维度
latent_dim = 64  # 自编码器学习的低维表示
batch_size = 256  # 批处理大小

class VariableLengthDataset(Dataset):
    def __init__(self, trajectories):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def prepare_data():
    dataset, _ = d3rlpy.datasets.get_dataset(args.dataset)
    dataset._buffer = dataset._buffer[:]

    traj_num = len(dataset._buffer)
    action_size = dataset._dataset_info.action_signature.shape[0][0]
    obs_size = dataset._dataset_info.observation_signature.shape[0][0]

    print(f"Number of trajectories: {traj_num}")
    all_traj_repsentations = []
    traj_labels = []
    for i in tqdm(range(traj_num)):
        obs= torch.tensor(dataset._buffer[i][0].observations)
        action = torch.tensor(dataset._buffer[i][0].actions)
        reward = torch.tensor(dataset._buffer[i][0].rewards)
        # import pdb; pdb.set_trace()
        all_traj_repsentations.append(torch.cat((obs, action,reward), dim=-1))
    dataset = VariableLengthDataset(all_traj_repsentations)

    # 数据集划分
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: nn.utils.rnn.pad_sequence(x, batch_first=True))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: nn.utils.rnn.pad_sequence(x, batch_first=True))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: nn.utils.rnn.pad_sequence(x, batch_first=True))
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = prepare_data()


# 初始化模型、损失函数和优化器
if args.model_type == "vae":
    model = VAE(input_dim, hidden_dim, latent_dim)
elif args.model_type == "autoencoder":
    model = AutoEncoder(input_dim, hidden_dim, latent_dim)
    
reconstruction_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.LinearLR(optimizer)

# 生成示例数据
# VAE的损失函数，包括重建损失和KL散度损失
def loss_function(reconstructed, x, mu, logvar, mask=None):
    if mask!=None:
        BCE = reconstruction_loss(reconstructed * mask, x * mask)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        BCE = reconstruction_loss(reconstructed, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD



# 训练变分自编码器
if args.train_from_scratch:
    num_epochs = 200
    model.to(args.device)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            
            batch_trajectories = batch.to(args.device)

            mask = (batch_trajectories != 0).float().to(args.device)
            
            optimizer.zero_grad()
            if args.model_type == "vae":
                reconstructed, mu, logvar = model(batch_trajectories)
                loss = loss_function(reconstructed, batch_trajectories, mu, logvar,mask)
            elif args.model_type == "autoencoder":
                reconstructed = model(batch_trajectories)
                loss = reconstruction_loss(reconstructed*mask, batch_trajectories*mask)
                # import pdb; pdb.set_trace()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            
            train_loss += loss.item() * batch_trajectories.size(0)

        train_loss /= len(train_loader.dataset)
        
        # 验证部分
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_trajectories = batch.to(args.device)
                mask = (batch_trajectories != 0).float().to(args.device)
                if args.model_type == "vae":
                    reconstructed, mu, logvar = model(batch_trajectories)
                    loss = loss_function(reconstructed, batch_trajectories, mu, logvar,mask)
                elif args.model_type == "autoencoder":
                    reconstructed = model(batch_trajectories)
                    loss = reconstruction_loss(reconstructed*mask, batch_trajectories*mask)
                val_loss += loss.item() * batch_trajectories.size(0)
        
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, f"./temp_result/{args.model_name}_best.pt")
        # 使用wandb记录损失
        if args.use_wandb:
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}',flush=True)

    print("Training Done!")

    # 测试部分
    model.eval()
    test_loss = 0
    model = torch.load(f"./temp_result/{args.model_name}_best.pt")
    with torch.no_grad():
        for batch in test_loader:
            batch_trajectories = batch.to(args.device)
            mask = (batch_trajectories != 0).float().to(args.device)
            if args.model_type == "vae":
                reconstructed, mu, logvar = model(batch_trajectories)
                loss = loss_function(reconstructed, batch_trajectories, mu, logvar,mask)
            elif args.model_type == "autoencoder":
                reconstructed = model(batch_trajectories)
                loss = reconstruction_loss(reconstructed*mask, batch_trajectories*mask)
            test_loss += loss.item() * batch_trajectories.size(0)

    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}',flush=True)

else:
    model = torch.load(f"./temp_result/{args.model_name}_best.pt")
    model.to(args.device)
    for batch in test_loader:
        batch_trajectories = batch.to(args.device)
        encodings = model.encode(batch_trajectories)
        print(encodings[0])
        import pdb; pdb.set_trace()


