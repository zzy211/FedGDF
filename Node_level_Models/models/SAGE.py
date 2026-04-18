#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Node_level_Models.helpers.func_utils import accuracy
from copy import deepcopy
from torch_geometric.nn import SAGEConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import f1_score

class GraphSage(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=None,use_prompt=False):

        super(GraphSage, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(nfeat, nhid))
        for _ in range(layer-2):
            self.convs.append(SAGEConv(nhid,nhid))
        self.gc2 = SAGEConv(nhid, nclass)
        self.dropout = 0.2
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None 
        self.weight_decay = weight_decay
        self.use_prompt = use_prompt

        if self.use_prompt:
            self.prompt_generator = nn.Sequential(
                nn.Linear(nhid, nhid),
                nn.LayerNorm(nhid),
                nn.ReLU(),
            )
            # self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)

    def forward(self, x, edge_index, edge_weight=None, node_types=None):
        layer_output = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
            layer_output.append(x)

        if(self.use_prompt):
            x = x * self.prompt_generator(x)

        x = self.gc2(x, edge_index)
        layer_output.append(x)
        return F.log_softmax(x,dim=1), layer_output[0], x
    
    def get_embeddings(self, x, edge_index, edge_weight=None, labels=None, args=None):
        x1 = F.relu(self.convs[0](x, edge_index))

        local_proto_label = {}  # Key is label, value is proto (a list)
        if args.alg_method == "FedTug":
            for node_idx in range(x.shape[0]):
                node_emb = torch.flatten(x1[node_idx])
                node_label = labels[node_idx]
                if node_label.item() not in local_proto_label.keys():
                    local_proto_label[node_label.item()] = [node_emb]
                else:
                    local_proto_label[node_label.item()].append(node_emb)
        return x1, local_proto_label
    
    def forward_logits(self, x, edge_index, edge_weight=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return x
    
    def rep_forward(self, x, edge_index, edge_weight=None):
        output = self.gc2(x, edge_index,edge_weight)
        return F.log_softmax(output,dim=1)
    

    def get_h(self, x, edge_index):

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        
        return x

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
            if args.alg_method == "FedSHA" or args.alg_method == 'FGPL' or args.alg_method == 'FedKD' or args.alg_method == 'MultiFedKD' or args.alg_method == 'FedKD_low_cost':
                return loss_train, local_proto_label, output_logits
            else:
                return loss_train, local_proto_label
        else:
            loss_train, loss_val, acc_train, acc_val = self._train_with_val(global_model,self.labels, idx_train, idx_val, train_iters, verbose,args)
            return loss_train, loss_val, acc_train, acc_val
    
    def _train_without_val(self, global_model, labels, idx_train, train_iters, verbose, agg_global_proto, args):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output, proto, output_logits = self.forward(self.features, self.edge_index, self.edge_weight)
            local_proto_label = {}  # Key is label, value is proto (a list)
            # 1. If using FedProto, compute local prototypes
            if args.alg_method == "Fedproto" or args.alg_method == "FedGH" or args.alg_method == "FGPL" or args.alg_method == "FedTug" or args.alg_method == "FedTGP":
                for node_id in idx_train:
                    node_emb = torch.flatten(proto[node_id])
                    node_label = labels[node_id]
                    if node_label.item() not in local_proto_label.keys():
                        local_proto_label[node_label.item()] = [node_emb]
                    else:
                        local_proto_label[node_label.item()].append(node_emb)
            # 2. Calculate loss       
            loss_train = F.nll_loss(output[idx_train], labels[idx_train].squeeze())
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
                    proto_new = torch.zeros_like(proto[idx_train])
                    for i, label in enumerate(labels[idx_train].squeeze()):
                        if label.item() in agg_global_proto:
                            proto_new[i] = agg_global_proto[label.item()]
                        else:
                            raise KeyError(f"{label.item()} not in agg_global_proto.keys()!")
                    loss_proto = F.mse_loss(proto_new, proto[idx_train])
                loss = loss_train + args.w_proto * loss_proto
            else:
                loss = loss_train

            loss.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        return loss.item(), local_proto_label, output_logits



    def _train_with_val(self,global_model, labels, idx_train, idx_val, train_iters, verbose, args):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = -10

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            if args.agg_method == "FedProx":
                # compute proximal_term
                proximal_term = 0.0
                for w, w_t in zip(self.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)

                loss_train = loss_train + (args.mu / 2) * proximal_term


            loss_train.backward()
            optimizer.step()



            self.eval()
            with torch.no_grad():
                output = self.forward(self.features, self.edge_index, self.edge_weight)
                loss_val = F.nll_loss(output[idx_val], labels[idx_val])
                acc_val = accuracy(output[idx_val], labels[idx_val])
                acc_train = accuracy(output[idx_train], labels[idx_train])
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print("acc_val: {:.4f}".format(acc_val))
            if acc_val >= best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
        print("acc_val",acc_val)
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        return loss_train.item(), loss_val.item(), acc_train, acc_val
    
    def train_with_logits(self, features,edge_index, edge_weight, global_score, labels, args, train_iters=5):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        criterion_kd = nn.KLDivLoss(reduction='batchmean')  # KL Divergence (Soft Loss)
        criterion_ce = nn.CrossEntropyLoss()  # Cross Entropy (Hard Loss)
        for i in range(train_iters):
            optimizer.zero_grad()

            student_logits = self.forward_logits(features, edge_index, edge_weight)
            student_probs = nn.functional.log_softmax(student_logits / args.TEMP, dim=1)
            loss_soft = criterion_kd(student_probs, global_score) * (args.TEMP ** 2)  # Temperature Scaling Compensation
            loss_hard = criterion_ce(student_logits, labels)
            loss = args.ALPHA * loss_soft + (1 - args.ALPHA) * loss_hard
            
            loss.backward()
            optimizer.step()  # Update model parameters
        
        return loss.item()


    def FedTAD_train_with_pseudo_graph(self, features, edge_index, edge_weight, each_class_idx, c, args, train_iters=2):
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
                loss_sem += loss_CE(local_pred[each_class_idx[class_i]], c[each_class_idx[class_i]])
            loss_sem.backward()
            optimizer.step()
        return loss_sem.item()
    
    def test_with_pseudo_graph(self, features, edge_index, edge_weight, labels, args):
        self.eval()
        with torch.no_grad():
            if args.fedtad_mode == 'rep_distill':
                local_pred = self.rep_forward(features, edge_index, edge_weight)
            else:
                local_pred, proto, output_logits = self.forward(features, edge_index, edge_weight)
            
            acc_test_pseudo = accuracy(local_pred, labels)
            return float(acc_test_pseudo)

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
        """Evaluate SAGE performance on test set using F1-score."""
        self.eval()
        with torch.no_grad():
            output, proto, output_logits = self.forward(features, edge_index, edge_weight)
            pred = output[idx_test].argmax(dim=1).cpu().numpy()
            true = labels[idx_test].squeeze().cpu().numpy()
            f1 = f1_score(true, pred, average='weighted')
        return float(f1)
    
    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels,idx_test):
        self.eval()
        output, proto, output_logits = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test]==labels[idx_test]).nonzero().flatten()
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return acc_test,correct_nids

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
            loss_sem = F.nll_loss(local_pred[idx_train], train_labels[idx_train])
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