from data_handling.speech_commands_data_module import SpeechCommandsDataModule
from model import LeNet, weight_init
import pytorch_lightning as pl


def train(args):
    data_module = SpeechCommandsDataModule(args)
    model = LeNet(args)
    model.apply(weight_init)

    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs)
    trainer.fit(model, data_module)





