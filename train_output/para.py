layer1 = ['transformer.decoder.layers.0.ca_text.in_proj_weight',
          'transformer.decoder.layers.0.ca_text.in_proj_bias',
          'transformer.decoder.layers.0.ca_text.out_proj.weight',
          'transformer.decoder.layers.0.ca_text.out_proj.bias',
          'transformer.decoder.layers.0.catext_norm.weight',
          'transformer.decoder.layers.0.catext_norm.bias']
layer2 = ['transformer.decoder.layers.1.ca_text.in_proj_weight',
          'transformer.decoder.layers.1.ca_text.in_proj_bias',
          'transformer.decoder.layers.1.ca_text.out_proj.weight',
          'transformer.decoder.layers.1.ca_text.out_proj.bias',
          'transformer.decoder.layers.1.catext_norm.weight',
          'transformer.decoder.layers.1.catext_norm.bias']
layer3 = ['transformer.decoder.layers.2.ca_text.in_proj_weight',
          'transformer.decoder.layers.2.ca_text.in_proj_bias',
          'transformer.decoder.layers.2.ca_text.out_proj.weight',
          'transformer.decoder.layers.2.ca_text.out_proj.bias',
          'transformer.decoder.layers.2.catext_norm.weight',
          'transformer.decoder.layers.2.catext_norm.bias']
layer4 = ['transformer.decoder.layers.3.ca_text.in_proj_weight',
          'transformer.decoder.layers.3.ca_text.in_proj_bias',
          'transformer.decoder.layers.3.ca_text.out_proj.weight',
          'transformer.decoder.layers.3.ca_text.out_proj.bias',
          'transformer.decoder.layers.3.catext_norm.weight',
          'transformer.decoder.layers.3.catext_norm.bias']
layer5 = ['transformer.decoder.layers.4.ca_text.in_proj_weight',
          'transformer.decoder.layers.4.ca_text.in_proj_bias',
          'transformer.decoder.layers.4.ca_text.out_proj.weight',
          'transformer.decoder.layers.4.ca_text.out_proj.bias',
          'transformer.decoder.layers.4.catext_norm.weight',
          'transformer.decoder.layers.4.catext_norm.bias']
layer6 = ['transformer.decoder.layers.5.ca_text.in_proj_weight',
          'transformer.decoder.layers.5.ca_text.in_proj_bias',
          'transformer.decoder.layers.5.ca_text.out_proj.weight',
          'transformer.decoder.layers.5.ca_text.out_proj.bias',
          'transformer.decoder.layers.5.catext_norm.weight',
          'transformer.decoder.layers.5.catext_norm.bias']
import torch

def load_full_model(model_name):
    model = torch.load(model_name)
    model['model']['transformer.decoder.layers.0.ca_support.in_proj_weight'] = model['model'].pop(layer1[0])
    model['model']['transformer.decoder.layers.0.ca_support.in_proj_bias'] = model['model'].pop(layer1[1])
    model['model']['transformer.decoder.layers.0.ca_support.out_proj.weight'] = model['model'].pop(layer1[2])
    model['model']['transformer.decoder.layers.0.ca_support.out_proj.bias'] = model['model'].pop(layer1[3])
    model['model']['transformer.decoder.layers.0.ca_support_norm.weight'] = model['model'].pop(layer1[4])
    model['model']['transformer.decoder.layers.0.ca_support_norm.bias'] = model['model'].pop(layer1[5])
    model['model']['transformer.decoder.layers.1.ca_support.in_proj_weight'] = model['model'].pop(layer2[0])
    model['model']['transformer.decoder.layers.1.ca_support.in_proj_bias'] = model['model'].pop(layer2[1])
    model['model']['transformer.decoder.layers.1.ca_support.out_proj.weight'] = model['model'].pop(layer2[2])
    model['model']['transformer.decoder.layers.1.ca_support.out_proj.bias'] = model['model'].pop(layer2[3])
    model['model']['transformer.decoder.layers.1.ca_support_norm.weight'] = model['model'].pop(layer2[4])
    model['model']['transformer.decoder.layers.1.ca_support_norm.bias'] = model['model'].pop(layer2[5])
    model['model']['transformer.decoder.layers.2.ca_support.in_proj_weight'] = model['model'].pop(layer3[0])
    model['model']['transformer.decoder.layers.2.ca_support.in_proj_bias'] = model['model'].pop(layer3[1])
    model['model']['transformer.decoder.layers.2.ca_support.out_proj.weight'] = model['model'].pop(layer3[2])
    model['model']['transformer.decoder.layers.2.ca_support.out_proj.bias'] = model['model'].pop(layer3[3])
    model['model']['transformer.decoder.layers.2.ca_support_norm.weight'] = model['model'].pop(layer3[4])
    model['model']['transformer.decoder.layers.2.ca_support_norm.bias'] = model['model'].pop(layer3[5])
    model['model']['transformer.decoder.layers.3.ca_support.in_proj_weight'] = model['model'].pop(layer4[0])
    model['model']['transformer.decoder.layers.3.ca_support.in_proj_bias'] = model['model'].pop(layer4[1])
    model['model']['transformer.decoder.layers.3.ca_support.out_proj.weight'] = model['model'].pop(layer4[2])
    model['model']['transformer.decoder.layers.3.ca_support.out_proj.bias'] = model['model'].pop(layer4[3])
    model['model']['transformer.decoder.layers.3.ca_support_norm.weight'] = model['model'].pop(layer4[4])
    model['model']['transformer.decoder.layers.3.ca_support_norm.bias'] = model['model'].pop(layer4[5])
    model['model']['transformer.decoder.layers.4.ca_support.in_proj_weight'] = model['model'].pop(layer5[0])
    model['model']['transformer.decoder.layers.4.ca_support.in_proj_bias'] = model['model'].pop(layer5[1])
    model['model']['transformer.decoder.layers.4.ca_support.out_proj.weight'] = model['model'].pop(layer5[2])
    model['model']['transformer.decoder.layers.4.ca_support.out_proj.bias'] = model['model'].pop(layer5[3])
    model['model']['transformer.decoder.layers.4.ca_support_norm.weight'] = model['model'].pop(layer5[4])
    model['model']['transformer.decoder.layers.4.ca_support_norm.bias'] = model['model'].pop(layer5[5])
    model['model']['transformer.decoder.layers.5.ca_support.in_proj_weight'] = model['model'].pop(layer6[0])
    model['model']['transformer.decoder.layers.5.ca_support.in_proj_bias'] = model['model'].pop(layer6[1])
    model['model']['transformer.decoder.layers.5.ca_support.out_proj.weight'] = model['model'].pop(layer6[2])
    model['model']['transformer.decoder.layers.5.ca_support.out_proj.bias'] = model['model'].pop(layer6[3])
    model['model']['transformer.decoder.layers.5.ca_support_norm.weight'] = model['model'].pop(layer6[4])
    model['model']['transformer.decoder.layers.5.ca_support_norm.bias'] = model['model'].pop(layer6[5])
    torch.save(model, 'para_changed_'+model_name)
def load_state(model_name):
    model = torch.load(model_name)
    model['transformer.decoder.layers.0.ca_support.in_proj_weight'] = model.pop(layer1[0])
    model['transformer.decoder.layers.0.ca_support.in_proj_bias'] = model.pop(layer1[1])
    model['transformer.decoder.layers.0.ca_support.out_proj.weight'] = model.pop(layer1[2])
    model['transformer.decoder.layers.0.ca_support.out_proj.bias'] = model.pop(layer1[3])
    model['transformer.decoder.layers.0.ca_support_norm.weight'] = model.pop(layer1[4])
    model['transformer.decoder.layers.0.ca_support_norm.bias'] = model.pop(layer1[5])
    model['transformer.decoder.layers.1.ca_support.in_proj_weight'] = model.pop(layer2[0])
    model['transformer.decoder.layers.1.ca_support.in_proj_bias'] = model.pop(layer2[1])
    model['transformer.decoder.layers.1.ca_support.out_proj.weight'] = model.pop(layer2[2])
    model['transformer.decoder.layers.1.ca_support.out_proj.bias'] = model.pop(layer2[3])
    model['transformer.decoder.layers.1.ca_support_norm.weight'] = model.pop(layer2[4])
    model['transformer.decoder.layers.1.ca_support_norm.bias'] = model.pop(layer2[5])
    model['transformer.decoder.layers.2.ca_support.in_proj_weight'] = model.pop(layer3[0])
    model['transformer.decoder.layers.2.ca_support.in_proj_bias'] = model.pop(layer3[1])
    model['transformer.decoder.layers.2.ca_support.out_proj.weight'] = model.pop(layer3[2])
    model['transformer.decoder.layers.2.ca_support.out_proj.bias'] = model.pop(layer3[3])
    model['transformer.decoder.layers.2.ca_support_norm.weight'] = model.pop(layer3[4])
    model['transformer.decoder.layers.2.ca_support_norm.bias'] = model.pop(layer3[5])
    model['transformer.decoder.layers.3.ca_support.in_proj_weight'] = model.pop(layer4[0])
    model['transformer.decoder.layers.3.ca_support.in_proj_bias'] = model.pop(layer4[1])
    model['transformer.decoder.layers.3.ca_support.out_proj.weight'] = model.pop(layer4[2])
    model['transformer.decoder.layers.3.ca_support.out_proj.bias'] = model.pop(layer4[3])
    model['transformer.decoder.layers.3.ca_support_norm.weight'] = model.pop(layer4[4])
    model['transformer.decoder.layers.3.ca_support_norm.bias'] = model.pop(layer4[5])
    model['transformer.decoder.layers.4.ca_support.in_proj_weight'] = model.pop(layer5[0])
    model['transformer.decoder.layers.4.ca_support.in_proj_bias'] = model.pop(layer5[1])
    model['transformer.decoder.layers.4.ca_support.out_proj.weight'] = model.pop(layer5[2])
    model['transformer.decoder.layers.4.ca_support.out_proj.bias'] = model.pop(layer5[3])
    model['transformer.decoder.layers.4.ca_support_norm.weight'] = model.pop(layer5[4])
    model['transformer.decoder.layers.4.ca_support_norm.bias'] = model.pop(layer5[5])
    model['transformer.decoder.layers.5.ca_support.in_proj_weight'] = model.pop(layer6[0])
    model['transformer.decoder.layers.5.ca_support.in_proj_bias'] = model.pop(layer6[1])
    model['transformer.decoder.layers.5.ca_support.out_proj.weight'] = model.pop(layer6[2])
    model['transformer.decoder.layers.5.ca_support.out_proj.bias'] = model.pop(layer6[3])
    model['transformer.decoder.layers.5.ca_support_norm.weight'] = model.pop(layer6[4])
    model['transformer.decoder.layers.5.ca_support_norm.bias'] = model.pop(layer6[5])
    torch.save(model, 'para_changed_'+model_name)
if __name__ == '__main__':
<<<<<<< HEAD
    model_name = 'checkpointnumlevel4_embedding_epoch9.pth'
=======
    model_name = 'epoch7_checkpointnumlevel4_embedding.pth'
>>>>>>> 7d782e1286011d0c6213fe15c7cf1e9bd4f10f4c
    load_full_model(model_name)
    #load_state(model_name)
















