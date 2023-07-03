# -----*----coding:utf8-----*----

import argparse
import torch
import pytorch_lightning as pl
from modeling.tplinker_plus.modeling_tplinker_plus import TplinkerPlusNer
from modeling.tplinker_plus.configure_tplinker_plus import TplinkerPlusNerConfig


class TplinerPlusNerModule(pl.LightningModule):
    
    def __init__(self, args):
        self.args=args
        config = TplinkerPlusNerConfig.from_pretrained(self.args.bert_model)
        
        self.model=TplinkerPlusNer.from_pretrained(args.bert_model,config=config)
        
        self.save_hyperparameters()


    @staticmethod
    def add_model_specific_args(parent_parser):
        model_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        model_parser.add_argument("--loss_type", choices=["bce", "dice"], default="bce", help="loss type")
        model_parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw", help="loss type")
        return model_parser

    def configure_optimizers(self):
        """Prepare optimizer and learning rate scheduler """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
                {
                        "params"      : [p for n, p in self.model.named_parameters() if
                                         not any(nd in n for nd in no_decay)],
                        "weight_decay": self.args.weight_decay,
                },
                {
                        "params"      : [p for n, p in self.model.named_parameters() if
                                         any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                },
        ]

        if self.optimizer == "adamw":
            optimizer = AdamW(
                    optimizer_grouped_parameters,
                    betas=(0.9, 0.999),  # according to RoBERTa paper
                    lr=self.args.lr,
                    eps=self.args.adam_epsilon,
            )
        else:
            # revisiting few-sample BERT Fine-tuning https://arxiv.org/pdf/2006.05987.pdf
            # https://github.com/asappresearch/revisit-bert-finetuning/blob/master/run_glue.py
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                          lr=self.args.lr,
                                          eps=self.args.adam_epsilon,
                                          weight_decay=self.args.weight_decay)

        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        # 注：只有在使用pytorch Lightning的LightningDataModule 时候才可以使用该方式回去训练集大小
        t_total = (len(self.trainer.datamodule.train_dataloader()) //
                   (self.trainer.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        warmup_steps = int(self.args.warmup_proportion * t_total)
        if self.args.lr_scheduler == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                            max_lr=self.args.lr,
                                                            pct_start=float(warmup_steps / t_total),
                                                            final_div_factor=self.args.final_div_factor,
                                                            total_steps=t_total,
                                                            anneal_strategy='linear')
        elif self.args.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=t_total)
        elif self.args.lr_scheduler == "polydecay":
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, t_total,
                                                                  lr_end=self.args.lr / 4.0)
        else:
            raise ValueError("lr_scheduler does not exist.")
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, attention_mask, token_type_ids):
        """forward inputs to BERT models."""
        return self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        start_labels = batch["start_position"]
        end_labels = batch["end_position"]
        label_mask = batch["label_mask"]
        start_logits, end_logits = self(input_ids, attention_mask, token_type_ids)
        total_loss, start_loss, end_loss = self.compute_loss(start_logits, end_logits, start_labels, end_labels,
                                                             label_mask)

        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        self.log("train_start_loss", start_loss, prog_bar=True)
        self.log("train_end_loss", end_loss, prog_bar=True)
        self.log("train_total_loss", total_loss, prog_bar=True)

        return total_loss























if __name__ == "__main__":
    
    parser=argparse.ArgumentParser(description="train tplinker ner model")
    parser.add_argument("--output_dir", type=str, default="./output_dir/", help="")
    parser.add_argument("--data_dir", type=str,
                        default="/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_code_repository/NlpTaskSpace/data/tplinker",
                        help="data dir")
    parser.add_argument("--bert_dir", type=str,
                        default="/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model",
                        help="bert config dir")
    parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained checkpoint path")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--lr_scheduler", choices=["linear", "onecycle", "polydecay"], default="onecycle")
    parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--warmup_proportion", default=0.1, type=int,
                        help="warmup steps used for scheduler.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--final_div_factor", type=float, default=1e4,
                        help="final div factor of linear decay scheduler")

    parser = pl.Trainer.add_argparse_args(parser)
    parser=TplinerPlusNerModule.add_model_specific_args(parser)

    args=parser.parse_args()
    


