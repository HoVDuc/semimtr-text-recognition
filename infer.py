from demo import *

args = {
    'config': 'configs/semimtr_finetune.yaml',
    'input': '../new_public_test/',
    'checkpoint': 'workdir/consistency-regularization/consistency-regularization_4_25000.pth',
    'device': -1
}

pt_outputs = main(args)
logging.info('Finished!')

with open('prediction.txt', 'a+') as f:
    for k, v in pt_outputs.items():
        f.write('{}\t{}\n'.format(os.path.basename(k), v))