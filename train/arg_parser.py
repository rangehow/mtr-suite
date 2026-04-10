import argparse




def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-backend',choices=['pytorch','triton'],default='triton')
    return parser.parse_args()

if __name__=='__main__':
    args =parse_args()