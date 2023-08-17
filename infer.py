from demo import *

args = {
    'config': 'configs/semimtr_finetune.yaml',
    'input': '../Datasets/Handwritten_OCR/test/',
    'checkpoint': 'workdir/consistency-regularization/best-consistency-regularization.pth',
    'device': 0
}

pt_outputs = main(args)
logging.info('Finished!')

with open('prediction.txt', 'a+') as f:
    for k, v in pt_outputs.items():
        f.write('{}\t{}\n'.format(os.path.basename(k), v))