#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Node_level_Models.helpers.func_utils import accuracy
from copy import deepcopy
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix

import torch.nn as nn
from sklearn.metrics import f1_score

class Lambda(nn.Module):
    """自定义Lambda层，封装任意函数"""
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=None,layer_norm_first=False,use_ln=False,use_prompt=False):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(layer-2):
            self.convs.append(GCNConv(nhid,nhid))
            self.lns.append(nn.LayerNorm(nhid))
        self.lns.append(nn.LayerNorm(nhid))
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None 
        self.weight_decay = weight_decay
        
        self.layer_norm_first = layer_norm_first    #是否在每一层前先进行归一化
        self.use_ln = use_ln
        self.use_prompt = use_prompt

        #如果是对特征重加权
        self.prompt = nn.Parameter(torch.randn(nclass, nhid))
        self.attn = nn.Linear(nhid, nclass)
        # if self.use_prompt:
        #     self.reset_parameters()       
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)
        torch.nn.init.xavier_uniform_(self.attn.weight)

    # def forward(self, x, edge_index, edge_weight=None):
    #     if(self.layer_norm_first):
    #         x = self.lns[0](x)  #首先对x进行归一化
    #     i=0
    #     #first conv
    #     x1 = F.relu(self.convs[0](x, edge_index, edge_weight))

    #     if self.use_ln:
    #         x = self.lns[i+1](x1)
    #         i += 1
    #         x = F.dropout(x, self.dropout, training=self.training)
    #     else:
    #         x = F.dropout(x1, self.dropout, training=self.training)
    #     #other conv
    #     for conv in self.convs[1:]:
    #         x = F.relu(conv(x, edge_index, edge_weight))
    #         if self.use_ln:
    #             x = self.lns[i+1](x)
    #         i+=1
    #         x = F.dropout(x, self.dropout, training=self.training)
    #     #cls layer
    #     x = self.gc2(x, edge_index,edge_weight)
    #     return F.log_softmax(x,dim=1), x1

    def forward(self, x, edge_index, edge_weight=None):
        
        layer_output = []
        if(self.layer_norm_first):
            x = self.lns[0](x)  #首先对x进行归一化
        i=0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
            layer_output.append(x)
            if self.use_ln:
                x = self.lns[i+1](x)
            i+=1
            #dropout以一定的概率随机丢弃一些神经元，不参与网络的前向传播和反向传播
            x = F.dropout(x, self.dropout, training=self.training)

        if self.use_prompt:
            cls_logits = self.gc2(x, edge_index, edge_weight)  # [n_nodes, nclass]
            # cls_probs = F.softmax(cls_logits, dim=1)           # [n_nodes, nclass]
            # weighted_prompts = torch.matmul(cls_probs, self.prompt)
            # x = x * weighted_prompts
            pred_classes = torch.argmax(cls_logits, dim=1)
            # print("pred_classes: ", pred_classes)
            x = x * self.prompt[pred_classes]
          
        x = self.gc2(x, edge_index,edge_weight)
        #最后返回的是x的log softmax的结果，表示每个节点属于每个类别的概率对数
        # return F.log_softmax(x,dim=1)
        return F.log_softmax(x,dim=1), layer_output[0], x
    
    def rep_forward(self, x, edge_index, edge_weight=None):
        #对传入的proto进行处理, 输入是layer_output[0]
        i = 0
        if self.use_ln:
            x = self.lns[i+1](x)
        i += 1
        x = F.dropout(x, self.dropout, training=self.training)
        for conv in self.convs[1:]:
            x = F.relu(conv(x, edge_index, edge_weight))
            if self.use_ln:
                x = self.lns[i+1](x)
            i += 1
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


    def forward_logits(self, x, edge_index, edge_weight=None):
        if(self.layer_norm_first):
            x = self.lns[0](x)  #首先对x进行归一化
        i=0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
            if self.use_ln:
                x = self.lns[i+1](x)
            i+=1
            #dropout以一定的概率随机丢弃一些神经元，不参与网络的前向传播和反向传播
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index,edge_weight)
        #最后返回的是x的log softmax的结果，表示每个节点属于每个类别的概率对数
        # return F.log_softmax(x,dim=1)
        return x
    


    def proto_forward(self, x, edge_index, edge_weight=None):
        if self.use_ln:
            x = self.lns[i+1](x)
            i += 1
            x = F.dropout(x, self.dropout, training=self.training)
        else:
            x = F.dropout(x, self.dropout, training=self.training)
        #other conv
        for conv in self.convs[1:]:
            x = F.relu(conv(x, edge_index, edge_weight))
            if self.use_ln:
                x = self.lns[i+1](x)
            i+=1
            x = F.dropout(x, self.dropout, training=self.training)
        #cls layer
        x = self.gc2(x, edge_index,edge_weight)
        return F.log_softmax(x,dim=1)

    
    # def rep_forward(self, x, edge_index, edge_weight=None):
    #     output = self.gc2(x, edge_index,edge_weight)
    #     return F.log_softmax(output,dim=1)


    def get_embeddings(self, x, edge_index, edge_weight=None, labels=None, args=None):
        if(self.layer_norm_first):
            x = self.lns[0](x)  #首先对x进行归一化
        #first conv
        x1 = F.relu(self.convs[0](x, edge_index, edge_weight))

        local_proto_label = {}  #key是label value是proto,是一个list
        if args.alg_method == "FedTug":
            for node_idx in range(x.shape[0]):
                node_emb = torch.flatten(x1[node_idx])
                node_label = labels[node_idx]
                if node_label.item() not in local_proto_label.keys():
                    local_proto_label[node_label.item()] = [node_emb]
                else:
                    local_proto_label[node_label.item()].append(node_emb)
        return x1, local_proto_label


    def fit(self, global_model, features, edge_index, edge_weight, labels, idx_train, agg_global_proto, args, idx_val=None, train_iters=10, verbose=False):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """

        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)

        if idx_val is None:
            loss_train, local_proto_label, output_logits = self._train_without_val(global_model, self.labels, idx_train, train_iters, verbose, agg_global_proto, args)
            if args.alg_method == 'FedSHA' or args.alg_method == 'FGPL' or args.alg_method == 'FedKD' or args.alg_method == 'MultiFedKD' or args.alg_method == 'FedKD_low_cost':
                return loss_train, local_proto_label, output_logits
            else:
                return loss_train, local_proto_label
        else:
            loss_train, loss_val, acc_train, acc_val = self._train_with_val(global_model, self.labels, idx_train, idx_val, train_iters, verbose,args)
            return  loss_train, loss_val, acc_train, acc_val

    def _train_without_val(self, global_model, labels, idx_train, train_iters, verbose, agg_global_proto, args):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        batch_count = 0

        for i in range(train_iters):
            optimizer.zero_grad()
            # 初始化全节点标签张量

            output, proto, output_logits = self.forward(self.features, self.edge_index, self.edge_weight)
            local_proto_label = {}  #key是label value是proto,是一个list
            #1.如果是fedproto计算本地原形
            if args.alg_method == "Fedproto" or args.alg_method == "FedGH" or args.alg_method == "FGPL" or args.alg_method == "FedTug" or args.alg_method == "FedTGP":
                for node_id in idx_train:
                    node_emb = torch.flatten(proto[node_id])
                    node_label = labels[node_id]
                    if node_label.item() not in local_proto_label.keys():
                        local_proto_label[node_label.item()] = [node_emb]
                    else:
                        local_proto_label[node_label.item()].append(node_emb)
            
            #2.计算loss        
            loss_train = F.nll_loss(output[idx_train], labels[idx_train].squeeze()) #把labels[idx_train]转化为一维的
            #2.1
            if args.alg_method == "FedProx":
                proximal_term = 0.0
                for w, w_t in zip(self.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss = loss_train + (args.mu / 2) * proximal_term
            #2.2 if fedproto, cal distance between local proto and global proto
            elif args.alg_method == "Fedproto" or args.alg_method == "FGPL" or args.alg_method == "FedTGP":
                if len(agg_global_proto)== 0:
                    loss_proto = 0.0
                else:
                    proto_new = torch.zeros_like(proto[idx_train])  # 只保留训练集部分
                    for i, label in enumerate(labels[idx_train].squeeze()):
                        if label.item() in agg_global_proto:
                            proto_new[i] = agg_global_proto[label.item()]
                        else:
                            raise KeyError(f"{label.item()} not in agg_global_proto.keys()!")
                    loss_proto = F.mse_loss(proto_new, proto[idx_train])# 确保形状匹配
                loss = loss_train + args.w_proto * loss_proto
            else:
                loss = loss_train
                         
            loss.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
            batch_count += 1
            
        return loss.item(), local_proto_label, output_logits



    def _train_with_val(self,global_model,labels, idx_train, idx_val, train_iters, verbose,args):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss_val = 100
        best_acc_val = -10
        
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output, proto, output_logits = self.forward(self.features, self.edge_index, self.edge_weight)
            
            if args.alg_method == "Fedins" and args.choose_node_method == "loss":
                per_node_loss = F.nll_loss(output[idx_train], labels[idx_train], reduction='none')
                for idx, loss_value in zip(idx_train, per_node_loss):
                    print(f'Node ID: {idx}, Loss: {loss_value.item()}')
                loss_train = per_node_loss.mean()
            else:
                loss_train = F.nll_loss(output[idx_train], labels[idx_train])

            if args.alg_method == "FedProx":
                # compute proximal_term
                proximal_term = 0.0
                for w, w_t in zip(self.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)

                loss_train = loss_train + (args.mu / 2) * proximal_term

            loss_train.backward()
            optimizer.step()



            self.eval()
            with torch.no_grad():
                output, proto, output_logits = self.forward(self.features, self.edge_index, self.edge_weight)
                loss_val = F.nll_loss(output[idx_val], labels[idx_val])
                acc_val = accuracy(output[idx_val], labels[idx_val])
                acc_train = accuracy(output[idx_train], labels[idx_train])


            
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print("acc_val: {:.4f}".format(acc_val))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

        return loss_train.item(), loss_val.item(), acc_train, acc_val

    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        with torch.no_grad():
            output, proto, output_logits = self.forward(features, edge_index, edge_weight)  
            acc_test = accuracy(output[idx_test], labels[idx_test].squeeze())
        return float(acc_test)
    
    def test_each_class(self, features, edge_index, edge_weight, labels, idx_test):
        """Evaluate GCN performance on test set and return accuracy per class."""
        self.eval()
        with torch.no_grad():
            output, proto, output_logits = self.forward(features, edge_index, edge_weight)
            test_output = output[idx_test]
            test_labels = labels[idx_test].squeeze()
            
            # 每个类别的精度
            preds = test_output.argmax(dim=1)
            unique_classes = torch.unique(test_labels)
            class_accuracies = {}
            for cls in unique_classes:
                cls_mask = (test_labels == cls)
                if cls_mask.any():
                    cls_acc = (preds[cls_mask] == test_labels[cls_mask]).float().mean().item()
                    class_accuracies[int(cls.item())] = cls_acc
                else:
                    class_accuracies[int(cls.item())] = 0.0
            
            return  class_accuracies

    def cal_f1_score(self, features, edge_index, edge_weight, labels, idx_test):
        """Evaluate GCN performance on test set using F1-score."""
        self.eval()
        with torch.no_grad():
            output, proto, output_logits = self.forward(features, edge_index, edge_weight)
            pred = output[idx_test].argmax(dim=1).cpu().numpy()
            true = labels[idx_test].squeeze().cpu().numpy()
            f1 = f1_score(true, pred, average='weighted')  # 可根据需求调整average参数
        return float(f1)
    
    def get_functional_embedding(self, random_graph):
        self.eval()
        features = random_graph.x
        edge_index = random_graph.edge_index
        edge_weight = random_graph.edge_weight
        with torch.no_grad():
            output, proto, output_logits = self.forward(features, edge_index, edge_weight)
            output = output.mean(dim=0)
            output = output.clone().detach().cpu().numpy()
        return output
            
    
    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels,idx_test):
        self.eval()
        output, proto, output_logits = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test]==labels[idx_test]).nonzero().flatten()   # return a tensor
        acc_test =accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        return acc_test,correct_nids
    

    
    def train_with_logits(self, features,edge_index, edge_weight, global_score, labels, args, train_iters=5):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        criterion_kd = nn.KLDivLoss(reduction='batchmean')  # KL散度（Soft Loss）
        criterion_ce = nn.CrossEntropyLoss()  # 交叉熵（Hard Loss）
        for i in range(train_iters):
            optimizer.zero_grad()



            student_logits = self.forward_logits(features, edge_index, edge_weight)
            student_probs = nn.functional.log_softmax(student_logits / args.TEMP, dim=1)
            #计算损失
            # 计算损失
            loss_soft = criterion_kd(student_probs, global_score) * (args.TEMP ** 2)  # 温度缩放补偿
            loss_hard = criterion_ce(student_logits, labels)
            loss = args.ALPHA * loss_soft + (1 - args.ALPHA) * loss_hard
            
            loss.backward()
            optimizer.step()  # 更新模型参数
        
        return loss.item()
    
    def train_with_pseudo_graph(self, features, edge_index, edge_weight, c, args, train_iters=2):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_CE = nn.CrossEntropyLoss().to(self.device)
        for i in range(train_iters):
            optimizer.zero_grad()
            loss_sem = 0
            if args.fedtad_mode == 'rep_distill':
                local_pred = self.rep_forward(features, edge_index, edge_weight)
            else:
                local_pred, proto, output_logits = self.forward(features, edge_index, edge_weight)
            for class_i in range(args.class_num):
                loss_sem += loss_CE(local_pred, c)
            loss_sem.backward()
            optimizer.step()
        return loss_sem.item()
    
    def train_with_pseudo_graph_serverl_layers(self, features, edge_index, edge_weight, c, args, train_iters=2):
        self.train()
        
        # 定义需要更新的参数组
        trainable_params = [
            {"params": self.lns[0].parameters()},
            {"params": self.convs[0].parameters()}
        ]
        
        # 其他参数不传入优化器（等效于冻结）
        optimizer = optim.Adam(
            trainable_params,  # 仅传入目标层参数
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        loss_CE = nn.CrossEntropyLoss().to(self.device)
        for i in range(train_iters):
            optimizer.zero_grad()
            loss_sem = 0
            if args.fedtad_mode == 'rep_distill':
                local_pred = self.rep_forward(features, edge_index, edge_weight)
            else:
                local_pred, proto, output_logits = self.forward(features, edge_index, edge_weight)
            
            for class_i in range(args.class_num):
                loss_sem += loss_CE(local_pred, c)
            
            loss_sem.backward()
            optimizer.step()
        
        return loss_sem.item()


    
    def FedTAD_train_with_pseudo_graph(self, features, edge_index, edge_weight, each_class_idx, c, args, train_iters=2):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # loss_CE = nn.CrossEntropyLoss().to(self.device)
        for i in range(train_iters):
            optimizer.zero_grad()
            loss_sem = 0
            if args.fedtad_mode == 'rep_distill':
                local_pred = self.rep_forward(features, edge_index, edge_weight)
            else:
                local_pred, proto, output_logits = self.forward(features, edge_index, edge_weight)
            for class_i in range(args.class_num):
                loss_sem += F.nll_loss(local_pred[each_class_idx[class_i]], c[each_class_idx[class_i]])
            loss_sem.backward()
            optimizer.step()
        return loss_sem.item()
    
    def FedTug_train_with_trigger(self, features, edge_index, edge_weight, idx_train, train_labels, train_iters=2, tri_index=None, selected_node=None, args=None):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # loss_CE = nn.CrossEntropyLoss().to(self.device)
        for i in range(train_iters):
            optimizer.zero_grad()
            if args.fedtug_mode == 'raw_distill':
                local_pred, proto, output_logits = self.forward(x=features, edge_index=edge_index, edge_weight=edge_weight)
            else:
                local_pred = self.rep_forward(x=features, edge_index=edge_index, edge_weight=edge_weight)
            # update_train_index = torch.cat([idx_train, tri_index], dim=0)
            # tri_labels_pred = torch.argmax(local_pred[selected_node], dim=1) 
            # tri_labels = torch.repeat_interleave(tri_labels_pred, repeats=args.trigger_size)
            # update_label = torch.cat([train_labels, tri_labels], dim=0)
           
            loss_sem = F.nll_loss(local_pred[idx_train], train_labels[idx_train])
            # loss_sem = F.nll_loss(local_pred[update_train_index], update_label[update_train_index])

            # target_tri_probs = torch.repeat_interleave(local_pred[selected_node], repeats=args.trigger_size, dim=0)
            # loss_kl = F.kl_div(local_pred[tri_index], target_tri_probs, reduction='mean', log_target=True)

            # loss = loss_sem + loss_kl
            loss = loss_sem

            loss.backward()
            optimizer.step()
        return loss.item()
    
    def FedTug_test_with_trigger(self, features, edge_index, edge_weight, idx_test, labels, args):
        self.eval()
        with torch.no_grad():
            if args.fedtug_mode == 'raw_distill':
                local_pred, proto, output_logits = self.forward(x=features, edge_index=edge_index, edge_weight=edge_weight)
            else:
                local_pred = self.rep_forward(x=features, edge_index=edge_index, edge_weight=edge_weight)
            acc_test = accuracy(local_pred[idx_test], labels[idx_test].squeeze())
        return float(acc_test)

            

    
    def test_with_pseudo_graph(self, features, edge_index, edge_weight, labels, args):
        self.eval()
        with torch.no_grad():
            if args.fedtad_mode == 'rep_distill':
                local_pred = self.rep_forward(features, edge_index, edge_weight)
            else:
                local_pred, proto, output_logits = self.forward(features, edge_index, edge_weight)
            
            acc_test_pseudo = accuracy(local_pred, labels)
            return float(acc_test_pseudo)
    


        



       
# %%
