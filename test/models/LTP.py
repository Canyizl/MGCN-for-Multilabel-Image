import torch
import torch.nn as nn

class MemoryGraph(nn.Module):
    def __init__(self,in_features, out_features, num_nodes):
        # B*1024*num_classes
        super(MemoryGraph, self).__init__()
        self.num_nodes = num_nodes

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv1= nn.Conv1d(in_features*2, in_features, 1)

        self.long_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.long_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.fc_eq3_w = nn.Linear(num_nodes, num_nodes)
        self.fc_eq3_u = nn.Linear(num_nodes, num_nodes)
        self.fc_eq4_w = nn.Linear(in_features, in_features)
        self.fc_eq4_u = nn.Linear(in_features, in_features)


    def forward_construct_short(self, x):
        ### Model global representations ###
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))

        ###  Conv  to the short memory ###
        x = torch.cat((x_glb, x), dim=1)
        x = self.conv1(x)
        x = torch.sigmoid(x)
        return x

    def forward_construct_long(self,x,short_memory):
        with torch.no_grad():
            long_a = self.long_adj(x.transpose(1, 2))
            long_a = long_a.view(-1, x.size(2))
            long_w = self.long_weight(short_memory)
            long_w = long_w.view(x.size(0) * x.size(2), -1)
        x_w = short_memory.view(x.size(0) * x.size(2), -1)  # B*num_c,1024 短期记忆包含全局关系，提取相对权重关系。生成weight
        x_a = x.view(-1, x.size(2))          # B*1024, num_c, 注意力直接，注重提取个体之间出现的关系。生成adj
        # eq(3)
        av = torch.tanh(self.fc_eq3_w(x_a) + self.fc_eq3_u(long_a))
        # eq(4)
        wv = torch.tanh(self.fc_eq4_w(x_w) + self.fc_eq4_u(long_w))
        # eq(5)
        x_a = x_a + av * long_a
        x_a = x_a.view(x.size(0),x.size(2),-1)
        x_w = x_w + wv * long_w
        x_w = x_w.view(x.size(0),x.size(1),x.size(2))
        long_adj = self.long_adj(x_a)
        long_weight = self.long_weight(x_w)
        x = x + short_memory
        long_graph_feature1 = torch.mul(long_adj.transpose(1, 2),x)
        long_graph_feature2 = torch.mul(long_graph_feature1,long_weight)
        long_graph_feature2 = torch.sigmoid(long_graph_feature2)
        return long_graph_feature2

    def forward(self,x):
        short_memory = self.forward_construct_short(x)
        long_memory = self.forward_construct_long(x,short_memory)
        return long_memory




class M_GCN(nn.Module):
    def __init__(self,model,num_classes):
        super(M_GCN, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.relu = nn.LeakyReLU(0.2)
        self.fc = nn.Conv2d(model.fc.in_features, num_classes, (1,1), bias=False)
        self.last_linear = nn.Conv1d(1024, self.num_classes, 1)
        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        self.conv_transform = nn.Conv2d(2048, 1024, (1,1))

        self.gcn = MemoryGraph(1024, 1024, num_classes)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward_feature(self, x):
        x = self.features(x)          #res101--> 2048
        return x

    def forward_sam(self, x):
        """ SAM module

        Shape:
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        mask = self.fc(x)                                           ##(B,C_in,H,W) -> (B,num_c,H,W)
        mask = mask.view(mask.size(0), mask.size(1), -1)            ##(B,num_c,H*W)
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)                                 ## mask = (B,H*W,num_c)

        x = self.conv_transform(x)                                  ##  (B,1024,H,W)
        x = x.view(x.size(0), x.size(1), -1)                        ## x = (B,1024,H*W)
        x = torch.matmul(x, mask)                                   ##  (B,1024,num_c)

        return x

    def forward_classification_ss(self, x):
        """ Get Semantic confident scores {s_s}.

        Shape:
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out) # C_out: num_classes
        """
        x = self.fc(x)                                  # (B,num_c,H,W)
        x = x.view(x.size(0), x.size(1), -1)            # (B,num_c,H*W)
        x = x.topk(1, dim=-1)[0].mean(dim=-1)
        return x

    def forward_gcn(self, x):
        x = self.gcn(x)
        return x

    def forward(self, x):
        x = self.forward_feature(x)
        out1 = self.forward_classification_ss(x)

        v = self.forward_sam(x)  # B*1024*num_classes
        z = self.forward_gcn(v)
        z = z + v
        out2 = self.last_linear(z)  # B*1*num_classes
        mask_mat = self.mask_mat.detach()
        out2 = (out2 * mask_mat).sum(-1)
        return out1, out2

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': large_lr_layers, 'lr': lr},
                ]

