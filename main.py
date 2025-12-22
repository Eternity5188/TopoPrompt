import sys, argparse, torch
sys.path.append("src/")

import numpy as np
import os.path as osp
import networkx as nx

from torch_geometric.loader import DataLoader

from src.models.model import TopoPrompt
from src.trainers.default_trainer import train, test_model
from utils.dataset import STDataset
from utils.data_convert import generate_samples
from utils.tools import mkdirs, load_test_best_model
from utils.initialize import init, seed_anything, init_log


def main(args):
    args.logger.info("Params: %s", vars(args))
    # Initialize result dictionary for metrics at different horizons
    args.result = {
        "3": {"MAE": {}, "MAPE": {}, "RMSE": {}},
        "6": {"MAE": {}, "MAPE": {}, "RMSE": {}},
        "12": {"MAE": {}, "MAPE": {}, "RMSE": {}},
        "Avg": {"MAE": {}, "MAPE": {}, "RMSE": {}}
    }
    mkdirs(args.save_data_path)
    vars(args)["graph_size_list"] = []

    # Iterate through each year from begin_year to end_year
    for year in range(args.begin_year, args.end_year + 1):
        # Load graph adjacency matrix for the current year
        graph = nx.from_numpy_array(np.load(osp.join(args.graph_path, str(year) + "_adj.npz"))["x"])
        
        if year == args.begin_year:
            vars(args)["init_graph_size"] = graph.number_of_nodes()
            vars(args)["subgraph"] = graph
            vars(args)["base_node_size"] = graph.number_of_nodes()
        else:
            vars(args)["init_graph_size"] = args.graph_size
        
        vars(args)["graph_size"] = graph.number_of_nodes()
        vars(args)["year"] = year
        args.graph_size_list.append(graph.number_of_nodes())
        
        # Load or preprocess data based on the data_process flag
        if args.data_process:
            inputs = generate_samples(
                31,
                osp.join(args.save_data_path, str(year)),
                np.load(osp.join(args.raw_data_path, str(year) + ".npz"))["x"],
                graph,
                val_test_mix=False
            )
        else:
            inputs = np.load(osp.join(args.save_data_path, str(year) + ".npz"), allow_pickle=True)
        
        args.logger.info("[*] Year {} loaded from {}.npz".format(args.year, osp.join(args.save_data_path, str(year))))
        
        # Normalize adjacency matrix and store as a PyTorch tensor
        adj = np.load(osp.join(args.graph_path, str(args.year) + "_adj.npz"))["x"]
        adj = adj / (np.sum(adj, axis=1, keepdims=True) + 1e-6)
        vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
        
        # If it's the first year and we are loading a pretrained model, skip training
        if year == args.begin_year and args.load_first_year:
            model, _ = load_test_best_model(args)
            test_loader = DataLoader(STDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
            test_model(model, args, test_loader, pin_memory=True)
            continue
        
        # Train or test based on configuration
        if args.train:
            train(inputs, args)
        elif args.auto_test:
            model, _ = load_test_best_model(args)
            test_loader = DataLoader(STDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=32)
            test_model(model, args, test_loader, pin_memory=True)
    
    # Print evaluation metrics across years for each prediction horizon
    args.logger.info("\n\nEvaluation Metrics Across Years:")
    for horizon in ["3", "6", "12", "Avg"]:
        for metric in ["MAE", "RMSE", "MAPE"]:
            info = ""
            values = []
            for year in range(args.begin_year, args.end_year + 1):
                if horizon in args.result and metric in args.result[horizon] and year in args.result[horizon][metric]:
                    val = args.result[horizon][metric][year]
                    info += "{:>10.2f}\t".format(val)
                    values.append(val)
            if values:
                avg_val = np.mean(values)
                args.logger.info("{:<4}\t{}\t{}".format(horizon, metric, info) + "\t{:>8.2f}".format(avg_val))

    # Print training time statistics
    total_time: float = 0.0
    args.logger.info("\nTraining Time Statistics:")
    for year in range(args.begin_year, args.end_year + 1):
        if year in args.result:
            total_t = args.result[year]["total_time"]
            avg_t = args.result[year]["average_time"]
            epochs = args.result[year]["epoch_num"]
            assert isinstance(total_t, (int, float)), f"Expected number, got {type(total_t)}"
            assert isinstance(avg_t, (int, float)), f"Expected number, got {type(avg_t)}"
            assert isinstance(epochs, int), f"Expected int, got {type(epochs)}"
            info = "Year {:<4} | Total Time: {:>10.4f}s | Avg/Epoch: {:>10.4f}s | Epochs: {}".format(
                year, total_t, avg_t, epochs
            )
            total_time += total_t
            args.logger.info(info)
    args.logger.info("Total training time across all years: {:.4f}s".format(total_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--conf", type=str, default="conf/PEMS4.json")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--paral", type=int, default=0)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--logname", type=str, default="info")
    parser.add_argument("--method", type=str, default="TopoPrompt")
    parser.add_argument("--load_first_year", type=int, default=0,
                        help="0: train first year from scratch, 1: load pretrained model for first year")
    parser.add_argument("--first_year_model_path", type=str,
                        default="log/PEMS3/topo-43/2011/14.7956.pkl",
                        help="Path to pretrained model for the first year")
    
    args = parser.parse_args()
    vars(args)["device"] = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != -1 else "cpu"
    vars(args)["methods"] = {'TopoPrompt': TopoPrompt}
    
    init(args)
    seed_anything(args.seed)
    init_log(args)
    
    main(args)