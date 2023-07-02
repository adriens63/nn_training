import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor



def get_head_bbox(in_features, num_classes, name):
    if name == 'fastrcnn':
        return FastRCNNPredictor(in_features, num_classes)

def get_head_segm(in_features, hidden_layer, num_classes, name):
    if name == 'maskrcnn':
        return MaskRCNNPredictor(in_features,
                            hidden_layer,
                            num_classes)


def get_model_instance(num_classes, hidden_layer_segm, heads):

    tasks = [task for task, name in heads.items() if name is not None]

    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    if  tasks == []:
        return model

    # bbox

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = get_head_bbox(in_features, num_classes, heads['bbox'])


    # segm

    # now get the number of input features for the mask classifier
    in_features_segm = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = hidden_layer_segm
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = get_head_segm(in_features_segm, hidden_layer, num_classes, heads['segm'])
    


    return model