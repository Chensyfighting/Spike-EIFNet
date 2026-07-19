from model.SpikeEIFNet import SpikeEIFNet


def build_model(model_name, num_classes, ohem, early_loss):
    if model_name == 'SpikeEIFNet':
        return SpikeEIFNet(classes=num_classes, ohem=ohem, augment=early_loss)
