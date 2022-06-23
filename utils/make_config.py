import argparse
import configparser


def write_config(name, args):
    config = configparser.ConfigParser()
    for k in args.keys():
        config['DEFAULT'][k] = str(args[k])

    with open(name, 'w') as configfile:
        config.write(configfile)


def read_config(name):
    config = configparser.ConfigParser()
    config.read(name)
    opt = dict()
    for k, v in list(config['DEFAULT'].items()):
        opt[k] = v
    opt = argparse.Namespace(**opt)
    return opt


import argparse
import json


def load_json(name):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='dummy')
    parser.add_argument('--port', type=str, default='dummy')
    with open(name, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)
    return args


def save_json(args, name):
    with open(name, 'wt') as f:
        json.dump(vars(args), f, indent=4)


if __name__ == '__main__':
    x = read_config('outputs/default.ini')