import segmentation_models_pytorch as smp

def init_model(args):
    
    if args.decoder.lower() == 'unet': 
        model = smp.Unet(args.encoder,in_channels=1,classes=args.out_class,activation=None,encoder_weights=args.pretrain_name)
    elif args.decoder.lower() == 'deeplabv3': 
        model = smp.DeepLabV3(args.encoder,in_channels=1,classes=args.out_class,activation=None,encoder_weights=args.pretrain_name)
    elif args.decoder.lower() == 'deeplabv3plus': 
        model = smp.DeepLabV3Plus(args.encoder,in_channels=1,classes=args.out_class,activation=None,encoder_weights=args.pretrain_name)
    elif args.decoder.lower() == 'pspnet': 
        model = smp.PSPNet(args.encoder,in_channels=1,classes=args.out_class,activation=None,encoder_weights=args.pretrain_name)
    elif args.decoder.lower() == 'pan': 
        model = smp.PAN(args.encoder,in_channels=1,classes=args.out_class,activation=None,encoder_weights=args.pretrain_name)
    return model