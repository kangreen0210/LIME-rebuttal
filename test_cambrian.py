from lmms_eval.models.cambrian_8b import *
from cambrian.model.builder import load_pretrained_model
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
pretrained='/ML-A100/team/mm/zhangge/models/cambrian_8b'
model_name = get_model_name_from_path(pretrained)
print(model_name)
tokenizer, model, image_processor, context_len = load_pretrained_model(pretrained, None, model_name)