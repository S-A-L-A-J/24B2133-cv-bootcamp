"""Train one named experiment. See README.md for the complete experiment suite."""
import argparse
from engine import Data, train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--method', choices=['ce', 'label_smoothing', 'gce', 'elr'], default='ce')
    parser.add_argument('--label', choices=['clean_label', 'noisy_label', 'synthetic_label'], default='noisy_label')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2133)
    parser.add_argument('--dynamics', action='store_true')
    args = parser.parse_args()
    train(Data(args.data), args.runs, args.run, args.method, args.label,
          args.seed, args.epochs, args.batch_size, dynamics=args.dynamics)
