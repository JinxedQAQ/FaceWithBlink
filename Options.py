import argparse


class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        ################******************** test settings ***************###########################

        self.parser.add_argument('--test_root', default='./0572_0019_0003/audio',
                                 help='path to videos or audios')
        self.parser.add_argument('--test_A_path', default='./demo_images',
                                 help='path input images')
        self.parser.add_argument('--test_resume_path', default='./checkpoints/405_0_Speech_reco_checkpoint.pth.tar',
                                 help='path to test resume models')
        self.parser.add_argument('--test_audio_video_length', type=int, default=99, help='# of files in the audio folder')
        self.parser.add_argument('--test_type', type=str, default='audio', help='type of data in the test root')
        self.parser.add_argument('--test_num', type=int, default=1, help='name of the result folder')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')

        ################******************** project settings ***************###########################

        self.parser.add_argument('--name', type=str, default='Speech_reco', help='The name of the model')
        self.parser.add_argument('--num_workers', default=8, type=int, help='# threads for loading data')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--feature_length', type=int, default=256, help='feature length')
        self.parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
        self.parser.add_argument('--label_size', type=int, default=500, help='number of labels for classification')
        self.parser.add_argument('--video_length', type=int, default=1, help='number of frames generate at each time step')
        self.parser.add_argument('--image_size', type=int, default=256, help='scale images to this size')
        self.parser.add_argument('--image_channel_size', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--mfcc_width', type=int, default=12, help='width of loaded mfcc feature')
        self.parser.add_argument('--mfcc_length', type=int, default=20, help='length of loaded mfcc feature')
        self.parser.add_argument('--blinkdata_width', type=int, default=3, help='blinkdata_width: total amount of current frame and frames before and after current frame')
        self.parser.add_argument('--image_block_name', type=str, default='align_face256', help='training folder name containing images')
        self.parser.add_argument('--blink_block_name', type=str, default='blinkdata', help='training folder name containing blinkdatas')
        self.parser.add_argument('--disfc_length', type=int, default=20, help='# of frames sending into the discriminate fc')
        self.parser.add_argument('--mul_gpu', type=bool, default=False, help='whether to use mul gpus')
        self.parser.add_argument('--cuda_on', type=bool, default=True, help='whether to use gpu')

        ################******************** training settings ***************###########################

        self.parser.add_argument('--main_PATH', type=str, default="/home/h2/lrwdatasetalign", help='path to datasets main path')
        self.parser.add_argument('--resume', type=bool, default=True, help='load pretrained model or not')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='where to save the checkpoints')
        self.parser.add_argument('--save_latest_freq', type=int, default=60000, help='how many steps to save the latest model')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay for adam')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=10000,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--num_epochs', type=int, default=100, help='# of epochs to run')
        self.parser.add_argument('--start_epoch', type=int, default=1, help='start epoch')
        self.parser.add_argument('--sequence_length', type=int, default=6, help='num of images used as gt during training')
        #self.parser.add_argument('--disfc_length', type=int, default=20, help='num of images used for disentanglement in pid')
        self.parser.add_argument('--pred_length', type=int, default=12, help='num of images used for classification')
        self.parser.add_argument('--id_label_size', type=int, default=40145, help='num of classes in the pretraining face pid space')

        ################******************** load pretraining settings ***************###########################
        #self.parser.add_argument('--resume', type=bool, default=True, help='whether to resume training')
        self.parser.add_argument('--resume_path', type=str, default='./checkpoints/101_DAVS_checkpoint.pth.tar',
                                 help='path to resume models')
        self.parser.add_argument('--resume_path_h', type=str, default='./checkpoints/405_21660000_Speech_reco_checkpoint.pth.tar',
                                 help='path to resume models by h')
        self.parser.add_argument('--resume_from_h', type=bool, default=True, help='if load model that was trained and saved by h')
        self.parser.add_argument('--id_pretrain_path', type=str, default='None',
                                 help='path to pid encoder resume models')

        ################********************** Loss settings **************############################

        self.parser.add_argument('--require_single_GAN', type=bool, default=False, help='whether to use GAN for single frame')
        self.parser.add_argument('--require_sequence_GAN', type=bool, default=True, help='whether to GAN for multiple frames')
        self.parser.add_argument('--output_68', type=bool, default=True, help='the layer')
        self.parser.add_argument('--no_lsgan', type=bool, default=True, help='whether to not use lsgan')
        self.parser.add_argument('--lambda_A', type=float, default=4, help='parameter for L1 loss')
        self.parser.add_argument('--lambda_B', type=float, default=8, help='parameter for L1 loss around the mouth')
        self.parser.add_argument('--lambda_CE', type=float, default=1, help='parameter for pid to wid cross entropy loss')
        self.parser.add_argument('--lambda_CE_inv', type=float, default=1000000, help='parameter for pid to wid inverse cross entropy loss')
        self.parser.add_argument('--L2margin', type=float, default=1, help='margin for the l2 contrastive loss')

        ################******************** visdom settings ***************###########################

        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display_freq', type=int, default=5000, help='display freq')
        self.parser.add_argument('--print_freq', type=int, default=8000, help='print freq')
        self.parser.add_argument('--eval_freq', type=int, default=10000, help='eval freq')#original==20
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                                 help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        
        self.parser.add_argument('--isTrain', type=bool, default=True, help='whether is training status')
        self.parser.add_argument('--require_audio', type=bool, default=True, help='Ture to load mfcc when dataloader')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt


