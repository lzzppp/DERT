
all_type = ['MCAN', 'ABD-Net', 'RAG-Net', 'RAG-Net-PA']
def set_model(args):
    if args.model_type not in all_type:
        raise ValueError('No model of this type')
    if args.model_type == 'MCAN':
        from resnet_mcan.models import mcan_reid
        return mcan_reid(args)
    elif args.model_type == 'ABD-Net':
        from abd_net.models import mcan_reid
        return mcan_reid(args)
    elif args.model_type == 'RAG-Net':
        from rga_classifier.models import mcan_reid
        return mcan_reid(args)
    elif args.model_type == 'RAG-Net-PA':
        from rga_pair.models import mcan_reid
        return mcan_reid(args)