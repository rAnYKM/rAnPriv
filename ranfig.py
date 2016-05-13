# Project Name: rAnPrivGP
# Author: rAnYKM (Jiayi Chen)
#
#          ___          ____       _       __________
#    _____/   |  ____  / __ \_____(_)   __/ ____/ __ \
#   / ___/ /| | / __ \/ /_/ / ___/ / | / / / __/ /_/ /
#  / /  / ___ |/ / / / ____/ /  / /| |/ / /_/ / ____/
# /_/  /_/  |_/_/ /_/_/   /_/  /_/ |___/\____/_/
#
# Script Name: ranfig.py
# Date: May. 9, 2016


import os
import sys
import getopt
import ConfigParser as cp


DEFAULT_SNAP_DIR = ''
DEFAULT_DATA_DIR = ''
DEFAULT_OUT_DIR = ''


def make_ranfig(snap_dir=DEFAULT_SNAP_DIR, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR):
    computer_name = os.getenv('COMPUTERNAME')
    config = cp.RawConfigParser()
    config.add_section('FILE_DIR')
    config.set('FILE_DIR', 'SNAP_DIR', snap_dir)
    config.set('FILE_DIR', 'DATA_DIR', data_dir)
    config.set('FILE_DIR', 'OUT_DIR', out_dir)
    # Writing our configuration file to 'example.cfg'
    with open(computer_name + '.rfg', 'wb') as configfile:
        config.write(configfile)


def load_ranfig():
    computer_name = os.getenv('COMPUTERNAME')
    config = cp.RawConfigParser()
    config.read(computer_name + '.rfg')
    snap_dir = config.get('FILE_DIR', 'SNAP_DIR')
    data_dir = config.get('FILE_DIR', 'DATA_DIR')
    out_dir = config.get('FILE_DIR', 'OUT_DIR')
    custom_dir = {'SNAP': snap_dir, 'DATA': data_dir, 'OUT': out_dir}
    return custom_dir


def main(argv):
    snap_dir = DEFAULT_SNAP_DIR
    data_dir = DEFAULT_DATA_DIR
    out_dir = DEFAULT_OUT_DIR
    try:
        opts, args = getopt.getopt(argv, "hs:d:o:", ["sfile=", "dfile=", "ofile="])
    except getopt.GetoptError:
        print('ERROR: ranfig.py -s <snap_dir> -d <data_dir> -o <out_dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('ranfig.py -s <snap_dir> -d <data_dir> -o <out_dir>')
            sys.exit()
        elif opt in ("-s", "--sfile"):
            snap_dir = arg
        elif opt in ("-d", "--dfile"):
            data_dir = arg
        elif opt in ("-o", "--ofile"):
            out_dir = arg
    make_ranfig(snap_dir, data_dir, out_dir)
    print('Fin. By rAnYKM')


if __name__ == '__main__':
    main(sys.argv[1:])
