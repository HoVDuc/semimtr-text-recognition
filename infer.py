from demo import *
import parser

def infer(args):
    mode = args.mode
    args_ = {
        'config': 'configs/semimtr_finetune.yaml',
        'input': args.input,
        # 'checkpoint': ['../../consistency-regularization_1_36000.pth', './workdir/best-consistency-regularization-0.448.pth'],
        'checkpoint': './workdir/consistency-regularization/consistency-regularization_4_23000.pth',
        'model_eval': 'alignment',
        'cuda': 0
    }
    pt_outputs = main(args_)
    logging.info('Finished!')
    if mode:
        with open('prediction.txt', 'w+') as f:
            for k, v in pt_outputs.items():
                f.write('{}\t{}\n'.format(os.path.basename(k), v))
    else:           
        for k, v in pt_outputs.items():
            print(k, v)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='path to image')
    parser.add_argument('--mode', action='store_true', required=False,
                        help='path to image')
    args = parser.parse_args()

    infer(args)