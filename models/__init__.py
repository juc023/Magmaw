"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.model_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the images that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.

Now you can use the model class by specifying flag '--model dummy'.
See our template model class 'template_model.py' for more details.
"""

import importlib
import pdb
from models.base_model import BaseModel


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    input : 'JSCCOFDM'
    """
    model_filename = "models." + model_name + "_model" 
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'

    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls # 'jsccofdmmodel'

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    if opt.data_type == 'image':
        opt.model = 'JSCCOFDM'
    elif opt.data_type == 'video':
        opt.model = 'VideoJSCCOFDM'
    elif opt.data_type == 'speech':
        opt.model = 'SpeechJSCCOFDM'
        opt.checkpoints_dir = opt.speech_checkpoints_dir
        opt.img_name = opt.speech_name
    elif opt.data_type == 'text':
        opt.model = 'TextJSCCOFDM'
        opt.checkpoints_dir = opt.text_checkpoints_dir
        opt.img_name = opt.text_name
    else:
        Exception("opt.dataset_mode is wrong")
        
    model = find_model_using_name(opt.model)

    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)

    return instance


def set_model_configuration(opt):
    if opt.dataset_mode == 'UCF':    
        opt.n_layers_D                               = 4          # Number of layers in the discriminator. Only used with GAN loss
        opt.n_downsample                             = 3          # Downsample times
        opt.n_blocks                                 = 4          # Numebr of residual blocks
        opt.n_video_layers_D                         = 4          # Number of layers in the discriminator. Only used with GAN loss
        opt.n_video_downsample                       = 4          # Downsample times
        opt.n_video_blocks                           = 3          # Numebr of residual blocks

    elif opt.dataset_mode == 'Speech':
        opt.n_layers_D                               = 4          # Number of layers in the discriminator. Only used with GAN loss
        opt.n_downsample                             = 2          # Downsample times
        opt.n_blocks                                 = 4          # Numebr of residual blocks

def create_seperate_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    if opt.coding_rate == 1:
    
        opt.data_type = 'image'
        opt.dataset_mode = 'UCF'
        set_model_configuration(opt)
        c1_image_model = create_model(opt)
        c1_image_model.setup(opt)
        c1_image_model.eval()
        
        opt.data_type = 'video'
        opt.dataset_mode = 'UCF'
        set_model_configuration(opt) 
        c1_video_model = create_model(opt)
        global_step = c1_video_model.load_model(c1_video_model, opt)
        c1_video_model = c1_video_model.cuda()
        c1_video_model.eval()

        opt.data_type = 'speech'
        opt.dataset_mode = 'Speech' 
        set_model_configuration(opt)
        opt.checkpoints_dir = opt.speech_checkpoints_dir
        opt.img_name = opt.speech_name
        c1_speech_model = create_model(opt)
        c1_speech_model.setup(opt)
        c1_speech_model.eval()

        opt.data_type = 'text'
        opt.dataset_mode = 'Text'
        set_model_configuration(opt)
        opt.checkpoints_dir = opt.text_checkpoints_dir
        opt.img_name = opt.text_name
        c1_text_model = create_model(opt)
        c1_text_model.setup(opt)
        c1_text_model.set_pad_idx(opt.pad_idx)
        c1_text_model.eval()
        
        return c1_image_model, c1_video_model, c1_speech_model, c1_text_model
    elif opt.coding_rate == 2:
                
        opt.data_type = 'image'
        opt.dataset_mode = 'UCF'
        opt.checkpoints_dir = opt.image_checkpoints_dir
        c2_image_model = create_model(opt)
        c2_image_model.setup(opt)
        c2_image_model.eval()
        
        opt.data_type = 'video'
        opt.dataset_mode = 'UCF' 
        c2_video_model = create_model(opt)
        global_step = c2_video_model.load_model(c2_video_model, opt)
        c2_video_model = c2_video_model.cuda()
        c2_video_model.eval()

        opt.data_type = 'speech'
        opt.dataset_mode = 'Speech' 
        opt.checkpoints_dir = opt.speech_checkpoints_dir
        opt.img_name = opt.speech_name
        c2_speech_model = create_model(opt)
        c2_speech_model.setup(opt)
        c2_speech_model.eval()

        opt.data_type = 'text'
        opt.dataset_mode = 'Text' 
        opt.checkpoints_dir = opt.text_checkpoints_dir
        opt.img_name = opt.text_name
        c2_text_model = create_model(opt)
        c2_text_model.setup(opt)
        c2_text_model.set_pad_idx(opt.pad_idx)
        c2_text_model.eval()

        return c2_image_model, c2_video_model, c2_speech_model, c2_text_model
    
    else:
        raise exception('opt.coding_rate is wrong')