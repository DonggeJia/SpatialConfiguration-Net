import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
#os.chdir('Spatial-U-net for hand bone joint localization')

def main(config):
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    

    print(config)
        
    train_loader = get_loader(image_path=config.train_path,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train')
    valid_loader = get_loader(image_path=config.valid_path,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='valid')
    test_loader = get_loader(image_path=config.test_path,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='test')


    solver = Solver(config, train_loader, valid_loader, test_loader)
    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
#     parser = argparse.ArgumentParser()

#     # training hyper-parameters
#     parser.add_argument('--img_ch', type=int, default=1)
#     parser.add_argument('--output_ch', type=int, default=37)
#     parser.add_argument('--num_epochs', type=int, default=100)
#     parser.add_argument('--num_epochs_decay', type=int, default=70)
#     parser.add_argument('--batch_size', type=int, default=1)
#     parser.add_argument('--num_workers', type=int, default=0)
#     parser.add_argument('--lr', type=float, default=0.0002)
#     parser.add_argument('--beta1', type=float, default=0.5, help='momentum1 in Adam')
#     parser.add_argument('--beta2', type=float, default=0.999, help='momentum2 in Adam')
#     parser.add_argument('--log_step', type=int, default=2)
#     parser.add_argument('--val_step', type=int, default=2)
#     parser.add_argument('--mode', type=str, default='train', help='train or test')
#     parser.add_argument('--model_type', type=str, default='SCN', help='SCN')
#     parser.add_argument('--model_path', type=str, default='./models')
#     parser.add_argument('--train_path', type=str, default='./dataset/train_input/')
#     parser.add_argument('--valid_path', type=str, default='./dataset/valid_input/')
#     parser.add_argument('--test_path', type=str, default='./dataset/test_input/')
#     parser.add_argument('--result_path', type=str, default='./result/')
    
#     config = parser.parse_args()
#     main(config)

    class Config_:

        # training hyper-parameters
        img_ch = 1
        output_ch = 37
        num_epochs = 50
        num_epochs_decay = 7000
        batch_size = 16
        num_workers = 0
        lr = 1e-3
        beta1 = 0.5  # momentum1 in Adam
        beta2 = 0.999  # momentum2 in Adam

        mode = 'train'
        model_type = 'SCN' 
        model_path = './models'
        train_path = './dataset/train_input/'
        valid_path = './dataset/valid_input/'
        test_path = './dataset/test_input/'
        result_path = './result/'

    config = Config_()
    main(config)
