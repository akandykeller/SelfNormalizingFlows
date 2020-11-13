import sys
import argparse
from snf.experiments import (selfnorm_fc_mnist, exact_fc_mnist,
    selfnorm_cnn_mnist, exact_cnn_mnist,
    emerging_cnn_mnist, exponential_cnn_mnist,
    selfnorm_glow_mnist, conv1x1_glow_mnist,
    toydensity, timing, snf_timescaling)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', type=str, help='experiment name')

def main():
    args = parser.parse_args()
    module_name = 'snf.experiments.{}'.format(args.name)
    experiment = sys.modules[module_name]
    experiment.main()