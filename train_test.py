from pytorch_lightning.callbacks import ModelCheckpoint

from data_handling.speech_commands_data_module import SpeechCommandsDataModule
from models.classifier import Classifier, weight_init
import pytorch_lightning as pl


def train(args):
    data_module = SpeechCommandsDataModule(args)
    model = Classifier(args, data_module.classes)
    model.apply(weight_init)

    checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="validation_weighted_f1",
                                          filename="speech_commands-{epoch:02d}-{validation_weighted_f1:.3f}",
                                          mode="max")
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)


def test(args):
    data_module = SpeechCommandsDataModule(args)
    model = Classifier.load_from_checkpoint(args.model_checkpoint)
    trainer = pl.Trainer(accelerator='gpu', devices=1)
    trainer.test(model, datamodule=data_module)