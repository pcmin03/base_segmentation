import segmentation_models_pytorch as smp

def init_loss(args):
    if args.lossname.lower() == 'jaccard': 
        criterion = smp.losses.JaccardLoss(mode='binary', from_logits=False)
    elif args.lossname.lower() == 'dice': 
        criterion = smp.losses.DiceLoss(mode='binary', from_logits=False)
    elif args.lossname.lower() == 'bce':    
        criterion = smp.losses.SoftBCEWithLogitsLoss()

    return criterion