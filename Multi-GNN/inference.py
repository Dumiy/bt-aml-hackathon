import torch
import pandas as pd
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, auc, roc_auc_score, roc_curve, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from training import get_model
from torch_geometric.nn import to_hetero, summary
import wandb
import logging
import os
import sys
import time
import numpy as np

script_start = time.time()

def infer_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #define a model config dictionary and wandb logging at the same time
    wandb.init(
        mode="disabled" if args.testing else "online",
        project="your_proj_name",

        config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None
        }
    )

    config = wandb.config

    #set the transform if ego ids should be used
    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None

    #add the unique ids to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    #get the model
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)

    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')
    
    # if not (args.avg_tps or args.finetune):
    #     command = " ".join(sys.argv)
    #     name = ""
    #     name = '-'.join(name.split('-')[3:])
    #     args.unique_name = name

    logging.info("=> loading model checkpoint")
    checkpoint = torch.load(f'{data_config["paths"]["model_to_load"]}')
    # start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint)
    model.to(device)

    logging.info("=> loaded checkpoint (epoch {})".format(0))
    t1 = time.perf_counter()
    x,y = [], []
    if not args.reverse_mp:
        logging.info("Train")
        f1 = evaluate_homo(tr_loader, tr_inds, model, tr_data, device, args)
        x.append(f1[0])
        y.append(f1[1])
        logging.info("Valid")
        f1 = evaluate_homo(val_loader, val_inds, model, val_data, device, args)
        x.append(f1[0])
        y.append(f1[1])
        logging.info("Test")
        f1 = evaluate_homo(te_loader, te_inds, model, te_data, device, args)
        x.append(f1[0])
        y.append(f1[1])
    else:
        te_f1, te_prec, te_rec = evaluate_hetero(tr_loader, model, tr_inds, device, args)
    t2 = time.perf_counter()
    logging.info(f"Runned inference in {t2 - t1:.2f}s")
    pred = np.hstack(x)
    ground_truth = np.hstack(y)
    f1 = f1_score(ground_truth, pred)
    logging.info(f"F1 SCORE {f1}")
    accuracy = accuracy_score(ground_truth, pred)
    precision = precision_score(ground_truth, pred)
    recall = recall_score(ground_truth, pred)
    f1 = f1_score(ground_truth, pred)
    fpr, tpr, _ = roc_curve(ground_truth, pred)
    roc_auc = auc(fpr, tpr)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC: {roc_auc:.4f}')

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc-curve.png')

    cm = confusion_matrix(ground_truth, pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Non-Laundering', 'Laundering'],
                yticklabels=['Non-Laundering', 'Laundering'])
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusionmatrix.png')

    wandb.finish()