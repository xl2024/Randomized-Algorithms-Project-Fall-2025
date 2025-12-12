import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import os
import random
import numpy as np

from models.mlp_cifar import MLP_CIFAR
from utils.masking import Masker
from utils.data_loader import get_cifar_loaders
from core.lll_sampler import LLLResampler
from core.rigl_sampler import RigLSampler

def count_dead_neurons(masker):
    dead_count = 0
    ignored_inputs = 0
    total_neurons = 0
    for name, mask in masker.masks.items():
        if mask.dim() == 2:
            row_sums = mask.sum(dim=1)
            dead_in_layer = (row_sums == 0).sum().item()
            
            dead_count += dead_in_layer
            total_neurons += mask.shape[0]

            col_sums = mask.sum(dim=0)
            ignored_inputs += (col_sums == 0).sum().item()

    return dead_count, ignored_inputs, total_neurons

def main():
    parser = argparse.ArgumentParser(description='Sparse MLP')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--method', type=str, default='lll', choices=['lll', 'rigl', 'random', 'lrandom'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--sparsity', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--min_fan_in', type=int, default=1)
    parser.add_argument('--max_fan_in', type=int, default=None)
    parser.add_argument('--min_fan_out', type=int, default=1)
    parser.add_argument('--max_fan_out', type=int, default=None)
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    print(f"--- Seed set to {args.seed} ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Experiment: {args.dataset.upper()} | {args.method.upper()} | Sparsity: {args.sparsity} ---")

    # Data
    trainloader, testloader, num_classes = get_cifar_loaders(
        dataset_name=args.dataset, batch_size=args.batch_size, num_workers=2)

    # MLP
    print("Initializing MLP Model...")
    model = MLP_CIFAR(num_classes=num_classes).to(device)
    masker = Masker(model, density=1.0 - args.sparsity)

    # Sampler
    sampler = None
    total_steps = args.epochs * len(trainloader)
    max_rigl_steps = int(total_steps * 0.75)
    convergence_log = None
    if args.method in ['lll', 'lrandom']:
        if args.sparsity >= 0.98:
            convergence_log = f"logs/{args.dataset}_{args.method}_s{args.sparsity}_seed{args.seed}_convergence.csv"
            with open(convergence_log, "w") as f:
                f.write("Epoch,AvgLoops,MaxLoops\n")

        sampler = LLLResampler(masker, 
                               model,
                               args.method,
                               total_steps,
                               min_fan_in=args.min_fan_in, 
                               max_fan_in=args.max_fan_in,
                               min_fan_out=args.min_fan_out,
                               max_fan_out=args.max_fan_out)

    elif args.method in ['rigl', 'random']:
        sampler = RigLSampler(masker, model, args.method, total_steps)
    
    print(f"{args.method} initialized. Updates every {args.interval} steps.")
        
    # Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)  # Learning Rate Scheduler

    # logs
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    log_filename = f"logs/{args.dataset}_{args.method}_s{args.sparsity}_seed{args.seed}.txt"
    print(f"Logging to: {log_filename}")
    
    with open(log_filename, "w") as f:
        f.write("Epoch\tAccuracy\tDeadNeurons\tIgnoredNeurons\n")

    # training
    best_acc = 0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()  # training mode
        pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        loops_per_epoch = []
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            global_step += 1
            
            optimizer.zero_grad()  # Gradients = 0
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()  # Calculate new gradients
            
            if global_step < max_rigl_steps and global_step % args.interval == 0:
                frac_and_loops = sampler.step(global_step)
                if args.method in ['lll', 'lrandom']:
                    _, loops = frac_and_loops
                    loops_per_epoch.extend(loops)

            masker.apply_mask_to_gradients()
            optimizer.step()  # Update weights
            masker.apply_mask_to_weights()

            _, predicted = outputs.max(1)  # dimension 1 = the class dimension -> (max_score(1D tensor), index(1D tensor))
            acc = 100. * predicted.eq(targets).sum().item()/targets.size(0)
            pbar.set_postfix({'Acc': f"{acc:.1f}%"})

        if convergence_log and len(loops_per_epoch) > 0:
            avg_loops = sum(loops_per_epoch) / len(loops_per_epoch)
            max_loops = max(loops_per_epoch)
            with open(convergence_log, "a") as f:
                f.write(f"{epoch+1},{avg_loops:.2f},{max_loops}\n")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        dead, ignored, total_n = count_dead_neurons(masker)
        print(f"Val Acc: {acc:.2f}% | Dead(In): {dead}/{total_n} | Ignored(Out): {ignored}")
        
        with open(log_filename, "a") as f:  # Append
            f.write(f"{epoch+1}\t{acc:.2f}\t{dead}\t{ignored}\n")

        if acc > best_acc:
            best_acc = acc
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints')
            torch.save(model.state_dict(), f'./checkpoints/best_{args.method}_{args.dataset}_s{args.sparsity}_seed{args.seed}.pth')
            
        scheduler.step()

    print(f"Training Finished. Best Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()