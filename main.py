import os
import subprocess
import glob
import pandas as pd
import matplotlib.pyplot as plt
from subprocess import PIPE
from datetime import datetime
import seaborn as sns
import re
import numpy as np
from matplotlib.dates import DateFormatter, HourLocator
from logging import getLogger, config
from argparse import ArgumentParser
import enum

# Constants
LOGGER_SETTING_FILE_PATH='logger_config.ini'

# Logger
config.fileConfig(LOGGER_SETTING_FILE_PATH)
logger = getLogger(__name__)

# Utils
def parse_value_with_unit(str):
    res = re.findall('(\d+(?:\.\d+)?)(\D+)?', str)
    if len(res) == 0:
        return None
    (size, unit) = res[0]
    return float(size), unit

def parse_size_in_kb(size_str):
    res = parse_value_with_unit(size_str)
    if res is None:
        return None
    (value, unit) = res

    l_unit = unit.lower()
    if l_unit == 'b':
        return value / 1024
    elif l_unit == 'mb':
        return value * 1024
    else:
        return value

def parse_size_in_ms(time_str):
    res = parse_value_with_unit(time_str)
    if res is None:
        return None
    (value, unit) = res

    l_unit = unit.lower()
    if l_unit == 'ns':
        return value / 1000
    elif l_unit == 'ms':
        return value
    elif l_unit == 's':
        return value * 1000
    elif l_unit == 'm':
        return value * 1000 * 60
    elif l_unit == 'h':
        return value * 1000 * 60 * 60
    return value

def parse_header_info(str):
    res = re.findall('([^:]+):\s+(.+)', str)
    if len(res) == 0:
        return None
    return res[0]

def get_begining_of_day(dt64):
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    dt = datetime.utcfromtimestamp(ts)
    dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return np.datetime64(dt)

def prase_result(type, value):
    if type in ['inuse_space', 'alloc_space']:
        return parse_size_in_kb(value)
    elif type in ['cpu']:
        return parse_size_in_ms(value)
    # 'inuse_objects', 'alloc_objects', 'goroutine', 'threadcreate'
    return float(value)

def default_chart_title(type):
    if type in ['inuse_space', 'alloc_space', 'inuse_objects', 'alloc_objects']:
        return 'Change of heap ({})'.format(type)
    elif type in ['cpu']:
        return 'Change of process time ({})'.format(type)
    elif type == 'goroutine':
        return 'Change of number of Goroutines'
    elif type == 'threadcreate':
        return 'Change of new thread creation'
    return 'Change of {}'.format(type)

def get_yaxis_label(type):
    if type in ['inuse_space', 'alloc_space']:
        return 'size [kb]'
    elif type in ['cpu']:
        return 'time [ms]'
    return 'number'

# Execute go tool pprof command and convert result to dataframe
def generate_data(profiles_dir, binary_path, heap_type, kind):
    logger.info("Generate data: profiles dir={}, binary path={}, heap_type={}, kind={}".format(profiles_dir, binary_path, heap_type, kind))
    filepaths = glob.glob('{}/*'.format(profiles_dir))
    if kind != KindEnum.FLAT and kind != KindEnum.CUM:
        raise Exception('kind {} must be either "flat" or "cum"'.format(kind))

    # Build base command
    base_cmd = 'go tool pprof'
    if heap_type is not None:
        base_cmd += ' -{}'.format(heap_type)
    base_cmd += ' -top'
    if binary_path is not None:
        base_cmd += ' {}'.format(binary_path)

    data_type = None
    df = pd.DataFrame(columns=['FAKE'])
    for filepath in filepaths:
        cmd = '{} {}'.format(base_cmd, filepath)

        logger.debug('cmd={}'.format(cmd))
        proc = subprocess.run(cmd, cwd=profiles_dir, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        if proc.returncode != 0:
            filename = os.path.basename(filepath)
            logger.error('return error in {}: {}'.format(filename, proc.stderr))
            continue

        lines = proc.stdout.splitlines()
        # binary_name = lines[0] # File: [name]
        (_, type) = parse_header_info(lines[1])
        (_, time) = parse_header_info(lines[2])
        # columns = lines[4]
        results = lines[5:]
        if lines[4].startswith("Dropped") or lines[3].startswith("Duration:"):
            # columns = lines[5]
            results = lines[6:]

        if data_type == None:
            data_type = type
        elif data_type != type:
            logger.warning('type mismatch, previous type={}, current type={}'.format(data_type, type))

        parsed_time = datetime.strptime(time, "%b %d, %Y at %I:%M%p (%Z)")
        row_key = pd.to_datetime(parsed_time.isoformat(), format='%Y-%m-%d')
        df.loc[row_key] = 0
        for res in results:
            values = [ss for ss in res.split(' ') if len(ss) > 0]
            if len(values) < 6:
                raise Exception('wrong format: {}'.format(res))
            value_s = values[0] if kind == KindEnum.FLAT else values[3]
            value = prase_result(type, value_s)
            func_name = values[5]

            df.at[row_key, func_name] = value
    del df['FAKE']
    return df, data_type

# Generate chart from dataframe
def generate_chart(df, type, config):
    title = config.title if config.title is not None else default_chart_title(type)
    num_lines = config.num if config.num is not None else None
    output_path = os.path.abspath(config.output_path) if config.output_path is not None else os.path.abspath('./output.svg')
    logger.info('Generate chart title={}, records={}, lines={}, output={}'.format(title, len(df.index.values), num_lines, output_path))

    # Presetting
    sns.set()
    sns.set_style("whitegrid")
    sns.set_context("paper")

    indices = df.index.values
    last = df.loc[indices[-1]]
    last.sort_values(ascending=False)
    columns = last.index.values[:num_lines] if num_lines is not None else last.index.values
    logger.debug('columns: {}'.format(','.join(columns)))

    # Prepare canvas
    fig, ax = plt.subplots(figsize=(12, 10))
    data = df[columns]
    sns.lineplot(data=data, ax=ax)

    # View setting
    ax.xaxis.set_major_locator(HourLocator(interval=6))
    ax.xaxis.set_major_formatter(DateFormatter('%m-%d:%H'))
    plt.title(title)
    plt.xlim(get_begining_of_day(indices[0]), None)
    plt.ylim(config.ymin, config.ymax)
    plt.xlabel("Time")
    plt.ylabel(get_yaxis_label(type) )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, ncol=1)

    fig.savefig(output_path, bbox_inches='tight')
    logger.info('saved {}'.format(output_path))


def generate_image(config):
    profiles_dir_path = os.path.abspath(config.profiles_dir)
    binary_path = os.path.abspath(config.binary_path) if config.binary_path is not None else None
    (dataframe, type) = generate_data(profiles_dir_path, binary_path, config.heap_type, config.kind)
    generate_chart(dataframe, type, config)

# Arguments
class KindEnum(enum.IntEnum):
    FLAT = 1
    CUM = 2

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return KindEnum[s.upper()]
        except KeyError:
            return s

def parse_args():
    usage = 'Usage: python {} PROFILES_DIR [--out OUTPUT_NAME] [--binary PATH_OF_BINARY] [--num NUMBER_OF_LINES] [--type inuse_space/alloc_space/inuse_objects/alloc_objects] [--kind flat/cum] [--title title] [--ymin min] [--ymax max] [--help]'.format(os.path.basename(__file__))
    argparser = ArgumentParser(usage=usage)
    argparser.add_argument('dir', type=str, help='path of directory for profiles')
    argparser.add_argument('-o', '--out', type=str, required=False, help='name of output file')
    argparser.add_argument('-b', '--binary', type=str, required=False, help='path of binary for analysis target')
    argparser.add_argument('-t', '--type', type=str, required=False, help='type of data, available only if the type of profiles is heap')
    argparser.add_argument('-k', '--kind', type=KindEnum.argparse, choices=list(KindEnum), required=False, default=KindEnum.FLAT, help='flat or cum')
    argparser.add_argument('-n', '--num', type=int, required=False, help="number of lines to show")
    argparser.add_argument('--title', type=str, required=False, help="title of chart")
    argparser.add_argument('--ymin', type=float, required=False, help="min value of yaxis")
    argparser.add_argument('--ymax', type=float, required=False, help="max value of yaxis")

    args = argparser.parse_args()

    return args

# Config
class Config:
    profiles_dir = None
    binary_path = None
    output_path = None
    num = None
    heap_type = None
    kind = KindEnum.FLAT
    title = None
    ymin = None
    ymax = None

    def __init__(self, profiles_dir, binary_path=None, output=None, num=None, heap_type=None, kind=KindEnum.FLAT, title=None, ymin=None, ymax=None):
        self.profiles_dir = profiles_dir
        self.binary_path = binary_path
        self.output_path = output
        self.num = num
        self.heap_type = heap_type
        self.kind = kind
        self.title = title
        self.ymin = ymin
        self.ymax = ymax

    def __str__(self):
        return "Config(profiles_dir={}, binary_path={}, output_name={}, num={}, heap_type={}, kind={}, title={}, ymin={}, ymax={})".format(self.profiles_dir, self.binary_path, self.output_path, self.num, self.heap_type, self.kind, self.title, self.ymin, self.ymax)

def main():
    args = parse_args()
    cfg = Config(args.dir, binary_path=args.binary, output=args.out, num=args.num, heap_type=args.type, kind=args.kind, title=args.title, ymin=args.ymin, ymax=args.ymax)
    logger.debug('config: {}'.format(cfg))
    generate_image(cfg)

if __name__ == '__main__':
    main()
