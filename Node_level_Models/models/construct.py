from Node_level_Models.models.GCN import GCN
from Node_level_Models.models.GAT import GAT
from Node_level_Models.models.SAGE import GraphSage
from Node_level_Models.models.Classifier import Classifier
from Node_level_Models.models.FedTAD_generator import FedTAD_ConGenerator
from Node_level_Models.models.FedGEN_generator import FedGEN_ConGenerator
from Node_level_Models.models.FedKD_generator import FedKD_Generator
from Node_level_Models.models.MultiFedKD_generator import MultiFedKD_Generator
from Node_level_Models.models.FedKD_generator_single_class import FedKD_Generator_single_class
from Node_level_Models.models.Discriminator import FakeGraphDiscriminator
from torch_geometric.utils import to_networkx, from_networkx


def model_construct(args, model_name, data, device, nclass, hidden=None, dropout=None, layer=None):
    if(args.dataset == 'Reddit2'):
        use_ln = True
        layer_norm_first = False
    else:
        use_ln = False
        layer_norm_first = False
    
    if args.use_weight_prompt:
        use_prompt = True
    else:
        use_prompt = False

    if(model_name == 'GCN'):
        model = GCN(nfeat=data.x.shape[1],\
                    nhid=hidden,\
                    nclass= nclass,\
                    dropout = dropout,\
                    lr=args.train_lr,\
                    weight_decay=args.weight_decay,\
                    layer=layer,\
                    device=device,\
                    use_ln=use_ln,\
                    layer_norm_first=layer_norm_first,\
                    use_prompt=use_prompt)
    elif(model_name == 'GAT'):
        model = GAT(nfeat=data.x.shape[1],\
                    nhid=hidden,\
                    nclass=nclass,\
                    dropout=dropout,\
                    lr=args.train_lr,\
                    weight_decay=args.weight_decay,\
                    layer=layer,\
                    device=device,\
                    use_ln=use_ln,\
                    layer_norm_first=layer_norm_first,\
                    heads=3,\
                    use_prompt=use_prompt)
    elif(model_name == 'GraphSage'):
        model = GraphSage(nfeat=data.x.shape[1],\
                    nhid=hidden,\
                    nclass= nclass,\
                    dropout=dropout,\
                    lr=args.train_lr,\
                    weight_decay=args.weight_decay,\
                    layer=layer,\
                    device=device,\
                    use_prompt=use_prompt)
    elif(model_name == "FedTAD_ConGenerator"):
        model = FedTAD_ConGenerator(noise_dim = args.noise_dim,\
                                    feat_dim = args.hidden if args.fedtad_mode == 'rep_distill' else data.x.shape[1],\
                                    out_dim = nclass,\
                                    dropout = args.dropout)
    elif(model_name == "FedGEN_Generator"):
        model = FedGEN_ConGenerator(noise_dim = args.noise_dim,\
                                    hidden_dim = data.x.shape[1],\
                                    class_dim = nclass,\
                                    dropout = args.dropout)
    elif(model_name == "FedKD_Generator"):
        model = FedKD_Generator(noise_dim = args.noise_dim,\
                                feat_dim = args.hidden if args.fedtad_mode == 'rep_distill' else data.x.shape[1],\
                                out_dim = nclass,\
                                dropout = args.dropout,
                                args=args)
    elif(model_name == "Discriminator"):
        model = FakeGraphDiscriminator(nfeat=data.x.shape[1],\
                                       nhid=args.hidden,\
                                       dropout=args.dropout,\
                                       lr=args.lr_c,\
                                       weight_decay=args.weight_decay,\
                                       device=device)
    elif(model_name == "MultiFedKD_Generator"):
        model = MultiFedKD_Generator(noise_dim = args.noise_dim,\
                                     feat_dim = args.hidden if args.fedtad_mode == 'rep_distill' else data.x.shape[1],\
                                     out_dim = nclass,\
                                     dropout = args.dropout,\
                                     args = args,\
                                     k_generators = args.k_generators)
    elif(model_name == "FedKD_Generator_single_class"):
        model = FedKD_Generator_single_class(noise_dim=args.noise_dim,\
                                             feat_dim=args.hidden if args.fedtad_mode == 'rep_distill' else data.x.shape[1],
                                             out_dim=nclass,
                                             sample_num=args.sample_num//nclass,
                                             dropout=args.dropout)
    elif(model_name == "Classifier"):
        model = Classifier(nhid=args.hidden,
                           nclass=nclass,
                           lr=args.train_lr,
                           weight_decay=args.weight_decay,
                           device=device)
    else:
        raise ValueError(f"Not implement {model_name}")
    return model