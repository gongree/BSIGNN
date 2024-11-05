import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
# from torch_geometric.nn import global_mean_pool

class BSI(nn.Module):
    def __init__(self, num_nodes, time_lag_length, hidden_dim):
        super(BSI, self).__init__()
        self.lstm_modules = nn.ModuleList([nn.LSTM(input_size=time_lag_length, hidden_size=hidden_dim, batch_first=True) for _ in range(num_nodes)])
        self.fc_modules = nn.ModuleList([nn.Linear(hidden_dim, num_nodes) for _ in range(num_nodes)])
        self.L = time_lag_length

    def forward(self, x):
        batch_size, num_nodes, time_length = x.size()
        
        contributions = []  # [num_nodes, T-L]
        for i, (lstm, fc) in enumerate(zip(self.lstm_modules, self.fc_modules)):
            region_contributions = []
            for t in range(self.L, time_length):
                region_input = x[:, i, t - self.L:t].unsqueeze(1)  # [batch_size, 1, L]
                lstm_out, _ = lstm(region_input)  # [batch_size, hidden_dim]
                contribution = fc(lstm_out[:, -1, :])  # [batch_size, num_nodes]
                region_contributions.append(contribution.unsqueeze(1))  # [batch_size, 1, num_nodes]
            
            region_contributions = torch.cat(region_contributions, dim=1)  # [batch_size, time_length - L, num_nodes]
            contributions.append(region_contributions.unsqueeze(-1))  # [batch_size, time_length - L, num_nodes, 1]
        contributions = torch.cat(contributions, dim=-1)  # [batch_size, time_length - L, num_nodes, num_nodes]

        # Calculate causal graph G based on contributions (Granger Causality)
        G = torch.zeros((batch_size, num_nodes, num_nodes)).to(x.device)
        for t in range(self.L, time_length):
            G += contributions[:, t - self.L, :, :] / (x[:, :, t].unsqueeze(1)) # + 1e-6)  # Avoid division by zero
        G /= (time_length - self.L)
        
        # Calculate predicted X
        X_hat = torch.sum(contributions, dim=2) + 1e-6 # [batch_size, time_length - L, num_nodes] 
        return G, X_hat


class DirectedGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(DirectedGraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

        self.bn = nn.BatchNorm1d(out_features*3)

    def forward(self, H, G):

        b, n, n = G.shape
        N_e = H @ self.weight

        # Apply Laplacian normalization
        D_out = torch.diag_embed(G.sum(dim=-1) + 1e-6)
        D_in = torch.diag_embed(G.sum(dim=-2) + 1e-6)
        

        G_F = torch.zeros_like(G)
        G_Sin = torch.zeros_like(G)
        G_So = torch.zeros_like(G)

        G_F = (G + G.transpose(2,1))/2

        for k in range(n):
            G_k_i_G_k_j = G[:, k, :].unsqueeze(2) * G[:, k, :].unsqueeze(1)  # G_{k,i} * G_{k,j}
            sum_G_k_v = G[:, k, :].sum(dim=1, keepdim=True).unsqueeze(2)  # sum_v G_{k,v}
            G_Sin += G_k_i_G_k_j / sum_G_k_v

            G_i_k_G_j_k = G[:, :, k].unsqueeze(2) * G[:, :, k].unsqueeze(1)  # G_{i,k} * G_{j,k}
            sum_G_v_k = G[:, :, k].sum(dim=1, keepdim=True).unsqueeze(2)  # sum_v G_{v,k}
            G_So += G_i_k_G_j_k / sum_G_v_k


        # # Symmetric normalization for G_F
        # G_F = 0.5 * (torch.inverse(D_out) @ G + G.transpose(-2, -1) @ torch.inverse(D_in))
        # 역행렬 대신 solve 사용하여 G_F 계산
        # try:
        #     D_out_inv = torch.linalg.solve(D_out, torch.eye(D_out.size(-1), device=G.device))
        #     D_in_inv = torch.linalg.solve(D_in, torch.eye(D_in.size(-1), device=G.device))
        #     G_F = 0.5 * (D_out_inv @ G + G.transpose(-2, -1) @ D_in_inv)
        # except RuntimeError as e:
        #     tqdm.write(f"Inversion error: {e}")
        #     G_F = torch.zeros_like(G)  # 오류 발생 시 기본값 설정

        # Normalized second-order in-degree (G_Sin) and out-degree (G_So)
        # G_Sin = G.transpose(-2, -1) @ G / torch.clamp(G.transpose(-2, -1).sum(dim=-1, keepdim=True), min=1e-6)
        # G_So = G @ G.transpose(-2, -1) / torch.clamp(G.sum(dim=-1, keepdim=True), min=1e-6)
        

        H_out = torch.cat([
            F.relu(G_F @ N_e),
            F.relu(G_Sin @ N_e),
            F.relu(G_So @ N_e)
        ], dim=-1)

        # window2 여기 예외처리
        # try:
        #     # batch normalize
        #     batch_size, node_num, feature_dim = H_out.size()
        #     H_out = H_out.view(batch_size * node_num, feature_dim)
        #     H_out = self.bn(H_out)
        #     H_out = H_out.view(batch_size, node_num, feature_dim)  # Reshape back to [batch, node_num, out_features * 3]
        # except:
        #     tqdm(f"DGC BN {H_out.shape}")
        return H_out


class TopKPooling(nn.Module):
    def __init__(self, in_features, k):
        super(TopKPooling, self).__init__()
        self.k = k
        self.score_weight = nn.Parameter(torch.Tensor(in_features, 1))
        nn.init.xavier_uniform_(self.score_weight)  # Xavier 초기화

    def forward(self, H, G):
        # H: [batch, node, feature], G: [batch, node, node]
        
        scores = torch.matmul(H, self.score_weight)/torch.norm(self.score_weight, p=2)
        topk_scores, topk_indices = torch.topk(scores.squeeze(-1), self.k, dim=1)

        
        batch_size, num_nodes, _ = H.shape
        H_retained = torch.zeros(batch_size, self.k, H.shape[2], device=H.device)
        G_retained = torch.zeros(batch_size, self.k, self.k, device=G.device)
        
        for i in range(batch_size):
            idx = topk_indices[i]
            H_retained[i] = H[i, idx] * torch.sigmoid(topk_scores[i]).unsqueeze(-1)
            G_retained[i] = G[i, idx][:, idx]  # 선택된 노드에 대한 서브그래프 추출

        return H_retained, G_retained


class BSIGNN(nn.Module):
    def __init__(self, num_nodes, time_lag_length, input_dim, hidden_size, num_classes):
        super(BSIGNN, self).__init__()
        self.bsi = BSI(num_nodes, time_lag_length, hidden_size)

        self.dgc1 = DirectedGraphConvolution(input_dim,hidden_size)
        self.pool1 = TopKPooling(k=num_nodes//3, in_features=hidden_size*3)
        self.dgc2 = DirectedGraphConvolution(hidden_size*3,hidden_size)
        self.pool2 = TopKPooling(k=num_nodes//9, in_features=hidden_size*3)

        self.bn = nn.BatchNorm1d((num_nodes//9)*hidden_size*3)
        self.fc = nn.Linear((num_nodes//9)*hidden_size*3, num_classes)

        self.brain_network = None
        self.mse_loss = 0
        self.sparsity_loss = 0
        self.rank_loss = 0

    def forward(self, x):

        # BSI Module
        brain_network, predicted = self.bsi(x)
        self.brain_network = brain_network

        self.sparsity_loss = torch.norm(brain_network, p=2, dim=(1, 2)).mean()
        self.rank_loss = torch.norm(brain_network, p='nuc', dim=(1, 2)).mean()
        self.mse_loss = F.mse_loss(predicted, x[:, :, self.bsi.L:].permute(0, 2, 1))  


        # Directed Graph Convolution Layers
        out = self.dgc1(x, brain_network)
        out, g = self.pool1(out, brain_network) # ([32, 37, 192]), ([32, 37, 37])
        out = self.dgc2(out, g) # ([32, 37, 192]) 
        out, g = self.pool2(out, g) # ([32, 12, 192]),([32, 12, 12])

        # Flatten and Classify
        out = out.view(out.size(0), -1) # ([32, 2304])
        # window1 (여기 예외처리)
        # try:
        #     out = self.bn(out)
        # except:
        #     tqdm.write(f"BSIGNN BN {out.shape}")
        out = self.fc(out)
        return F.softmax(out, dim=1)

    def calculate_loss(self, y, logits,lambda_ce=1, lambda_sparse=1, lambda_rank=1, lambda_mse=1, weight=[0.6219, 0.3781]):

        # Classification Loss
        # classification_loss = F.cross_entropy(logits, y, weight=torch.tensor(weight, device=y.device))
        classification_loss = F.nll_loss(logits, y, weight=torch.tensor(weight, device=y.device))

        # tqdm.write(f"Loss : {classification_loss.item():.2f}, {self.sparsity_loss.item():.2f}, {self.rank_loss.item():.2f}, {self.mse_loss.item():.2f}")
        
        return classification_loss, self.sparsity_loss, self.rank_loss, self.mse_loss



