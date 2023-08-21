from demo import *

args = {
    'config': 'configs/semimtr_finetune.yaml',
    'input': '../Datasets/Handwritten_OCR/new_public_test/',
    'checkpoint': './best-consistency-regularization.pth',
    'model_eval': 'alignment',
    'cuda': 0
}

pt_outputs = main(args)
logging.info('Finished!')

with open('prediction.txt', 'a+') as f:
    for k, v in pt_outputs.items():
        f.write('{}\t{}\n'.format(os.path.basename(k), v))
        
# for k, v in pt_outputs.items():
#     print(k, v)