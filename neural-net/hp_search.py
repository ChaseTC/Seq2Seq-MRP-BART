from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, AutoConfig
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from pathlib import Path
import argparse
import yaml
import wandb

def hp_search(config):
    wandb.login(key=config.key)

    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'eval_loss',
            'goal': 'minimize'
        }
    }
    param_space = {
        'weight_decay': {
            'distribution': 'q_uniform',
            'min': 0.001,
            'max': 0.01,
            'q': 0.001
        },
        'learning_rate': {
            'values': [1e-5, 5e-5]
        },
        'dropout': {
            'distribution': 'q_uniform',
            'min': 0.1,
            'max': 0.25,
            'q': 0.05
        },
        'gradient_accumulation_steps': {
            'values': [1, 5, 10, 15, 20]
        }
        
    }
    sweep_config['parameters'] = param_space

    sweep_id = wandb.sweep(sweep_config, project=config.project)

    path = '/home/ctingchong/lustre/GCP/data/subset'
    dev_dataset = load_dataset('json', data_dir=path + '/dev')
    dataset = load_dataset('json', data_dir=path + '/train')
    dataset['dev'] = dev_dataset.pop('train')

    model_name = 'facebook/bart-base'

    base_config = AutoConfig.from_pretrained(model_name)
    base_config.no_repeat_ngram_size = 0
    base_config.prefix = " "
    base_config.output_attentions = True
    base_config.dropout = 0.25
    base_config.attention_dropout = 0.0

    tokenizer = BartTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens([":BV", ":BV-of", ":ARG4", ":ARG4-of", ":L-INDEX", ":L-INDEX-of", ":L-HNDL", ":L-HNDL-of", ":ARG1", ":ARG1-of", ":ARG2", ":ARG2-of", ":R-INDEX", ":R-INDEX-of", ":R-HNDL", ":R-HNDL-of", ":ARG", ":ARG-of", ":ARG3", ":ARG3-of",
                            "proper_q", "named", "compound", "_n_1", "udef_q", "def_explicit_q", "poss", "_n_of", "_v_to", "pron", "pronoun_q", "_v_1", "_q", "_a_1", "_p", "card", "_n_unknown", "with_p", "subord", "_p_state", "part_of", "focus_d",
                            "parg_d", "_q_dem", "_v_cause", "def_implicit_q", "mofy", "loc_nonsp", "measure", "_v_as", "_p_temp", "appos", "unknown", "_c", "nominalization", "id", "_n_in", "_v_id", "number_q", "_x_deg", "_v_for", "_n_temp", "ord",
                            "times", "_p_per", "time_n", "comp", "yofc", "_a_to", "much-many_a", "_pre-_a_ante", "neg", "_v_at", "thing", "which_q", "dofw", "generic_entity", "_v_modal", "_x_then", "_x_subord", "ellipsis_ref", "little-few_a", "_a_at-for-of",
                            "_x", "_n_of-n", "implicit_conj", "_v_in", "named_n", "_p_nbar", "_v_there", "_v_on", "_a_as", "comp_equal", "_v_back", "_a_for", "of_p", "dofm", "idiom_q_i", "_v_2", "_v_with", "_n_to", "superl", "_a_unknown", "place_n",
                            "every_q", "temp_loc_x", "_v_unknown", "_v_nv", "person", "_a_on", "_v_of", "_a_of", "_v_over", "season", "unspec_manner", "manner", "_n_on-about", "_v_about", "eventuality", "_n_of-on", "_x_h", "_p_dir", "comp_so",
                            "_p_namely", "_v_prd", "_v_up", "_n_of-to", "_p_means", "relative_mod", "_a_thus", "_n_of-about", "_a_with", "_a_about", "_v_name", "_v_from", "excl", "discourse", "_v_state", "_n_about", "abstr_deg", "_un-_a_rvrs",
                            "_n_for", "_re-_a_again", "_n_item", "_n_of-for", "_v_down", "temp", "_v_qmodal", "year_range", "comp_less", "_n_i", "_a_than-from", "_v_off", "plus", "fraction", "interval", "interval_p_start", "interval_p_end", "_high_a_1",
                            "_and_c", "_v_into", "_v_from-to", "parenthetical", "_v_out", "_a_from", "_p_time", "_a_at", "_long_a_1", "_n_cur", "elliptical_n", "_n_2", "free_relative_q", "_x_prd", "_a_also", "numbered_hour", "_a_in", "comp_too", "reason"])

    max_input_length = 128
    max_target_length = 512

    def preprocess(examples):
        inputs = [ex for ex in examples['sentence']]
        targets = [ex for ex in examples['penman']]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)
        
        model_inputs['labels'] = labels['input_ids']

        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    def train(config = None):
        with wandb.init(config=config):
            wb_config = wandb.config
            base_config.dropout = wb_config.dropout

            model = BartForConditionalGeneration(config=base_config)
            model.resize_token_embeddings(len(tokenizer))
            data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

            args = Seq2SeqTrainingArguments(
                output_dir='models/hp-search',
                evaluation_strategy='epoch',
                learning_rate=wb_config.learning_rate,
                per_device_train_batch_size=5,
                per_device_eval_batch_size=5,
                weight_decay=wb_config.weight_decay,
                gradient_accumulation_steps=wb_config.gradient_accumulation_steps,
                max_grad_norm=2.5,
                warmup_steps=1,
                save_strategy='no',
                num_train_epochs=5,
                report_to='wandb'
                )

            trainer = Seq2SeqTrainer(
                model,
                args,
                train_dataset=tokenized_dataset['train'],
                eval_dataset=tokenized_dataset['dev'],
                data_collator=data_collator,
                tokenizer=tokenizer,
            )

            trainer.train()

    wandb.agent(sweep_id, train, count=10)

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', type=Path, help='directory path to config', default='config/config.yaml')

    args = argparser.parse_args()

    with args.config.open() as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    hp_search(config)

if __name__ == '__main__':
    main()