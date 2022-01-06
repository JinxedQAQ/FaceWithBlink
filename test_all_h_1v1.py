import time
from Options_all import BaseOptions
from util import util
from util.visualizer import Visualizer
from torch.utils.data import DataLoader
import os
import ntpath


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

opt = BaseOptions().parse()
opt.test_type = 'audio'
#mfccs_root = '/media/h2/GMT7/Work/ForResults/mfccs/'
mfccs_root = './inputvideos/'
if opt.test_type == 'audio':
    mfccs_root = './inputaudios/'
audios = os.listdir(mfccs_root)
# path to audios
opt.test_root = mfccs_root + audios[0]
#opt.test_root = './0572_0019_0003/audio'

opt.test_audio_video_length = len(os.listdir(opt.test_root))-1
opt.test_A_path = './faces/1vs1'

if opt.test_type == 'audio':
    import Test_Gen_Models.Test_Audio_Model as Gen_Model
    from Dataloader.Test_load_audio import Test_VideoFolder
else:
    raise('test type select error')

opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.sequence_length = 1
#test_nums = [1, 2, 3, 4]  # choose input identity images
test_names = os.listdir(opt.test_A_path)

model = Gen_Model.GenModel(opt)
# _, _, start_epoch = util.load_test_checkpoint(opt.test_resume_path, model)
start_epoch = opt.start_epoch
visualizer = Visualizer(opt)
# find the checkpoint's path name without the 'checkpoint.pth.tar'
path_name = ntpath.basename(opt.test_resume_path)[:-19]
web_dir = os.path.join(opt.results_dir, path_name, '%s_%s' % ('test', start_epoch))
for ado in audios:
    opt.test_root = mfccs_root + ado
    opt.test_audio_video_length = len(os.listdir(opt.test_root))-1
    for i in test_names:
        if i.split('.')[0] != ado:
            continue
        web_dir = os.path.join(opt.results_dir, path_name, '%s_%s' % (ado.split('_')[-1].split('.')[0], i.split('.')[0]))
        A_path = os.path.join(opt.test_A_path, i)
        test_folder = Test_VideoFolder(root=opt.test_root, A_path=A_path, config=opt)
        test_dataloader = DataLoader(test_folder, batch_size=1,
                                    shuffle=False, num_workers=1)
        model, _, start_epoch = util.load_test_checkpoint(opt.test_resume_path, model)

        # inference during test

        for i2, data in enumerate(test_dataloader):
            if i2 < 5:
                model.set_test_input(data)
                model.test_train()

        # test
        start = time.time()
        for i3, data in enumerate(test_dataloader):
            model.set_test_input(data)
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            visualizer.save_images_test(web_dir, visuals, img_path, i3, opt.test_num)
        end = time.time()
        print('finish processing in %03f seconds' % (end - start))

