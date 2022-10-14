import argparse
import yaml
from pathlib import Path

from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, AutoConfig
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer


def train(config, args):
    if config['wandb']:
        import wandb
        wandb.login(key=config['key'])
        wandb.init(project=config['project'])

    data_path = 'data/subset'
    dev_dataset = load_dataset('json', data_dir=data_path + '/dev')
    dataset = load_dataset('json', data_dir=data_path + '/train')
    dataset['dev'] = dev_dataset.pop('train')

    model_type = config['model']

    model_config = AutoConfig.from_pretrained(model_type)
    model_config.no_repeat_ngram_size = 0
    model_config.prefix = " "
    model_config.output_attentions = True
    model_config.dropout = config['dropout']
    model_config.attention_dropout = config['attention_dropout']

    tokenizer = BartTokenizer.from_pretrained(model_type)
    tokenizer.add_tokens([":BV", ":BV-of", ":ARG4", ":ARG4-of", ":L-INDEX", ":L-INDEX-of", ":L-HNDL", ":L-HNDL-of", ":ARG1", ":ARG1-of", ":ARG2", ":ARG2-of", ":R-INDEX", ":R-INDEX-of", ":R-HNDL", ":R-HNDL-of", ":ARG", ":ARG-of", ":ARG3", ":ARG3-of",
                            "proper_q", "named", "compound", "_n_1", "udef_q", "def_explicit_q", "poss", "_n_of", "_v_to", "pron", "pronoun_q", "_v_1", "_q", "_a_1", "_p", "card", "_n_unknown", "with_p", "subord", "_p_state", "part_of", "focus_d",
                            "parg_d", "_q_dem", "_v_cause", "def_implicit_q", "mofy", "loc_nonsp", "measure", "_v_as", "_p_temp", "appos", "unknown", "_c", "nominalization", "id", "_n_in", "_v_id", "number_q", "_x_deg", "_v_for", "_n_temp", "ord",
                            "times", "_p_per", "time_n", "comp", "yofc", "_a_to", "much-many_a", "_pre-_a_ante", "neg", "_v_at", "thing", "which_q", "dofw", "generic_entity", "_v_modal", "_x_then", "_x_subord", "ellipsis_ref", "little-few_a", "_a_at-for-of",
                            "_x", "_n_of-n", "implicit_conj", "_v_in", "named_n", "_p_nbar", "_v_there", "_v_on", "_a_as", "comp_equal", "_v_back", "_a_for", "of_p", "dofm", "idiom_q_i", "_v_2", "_v_with", "_n_to", "superl", "_a_unknown", "place_n",
                            "every_q", "temp_loc_x", "_v_unknown", "_v_nv", "person", "_a_on", "_v_of", "_a_of", "_v_over", "season", "unspec_manner", "manner", "_n_on-about", "_v_about", "eventuality", "_n_of-on", "_x_h", "_p_dir", "comp_so",
                            "_p_namely", "_v_prd", "_v_up", "_n_of-to", "_p_means", "relative_mod", "_a_thus", "_n_of-about", "_a_with", "_a_about", "_v_name", "_v_from", "excl", "discourse", "_v_state", "_n_about", "abstr_deg", "_un-_a_rvrs",
                            "_n_for", "_re-_a_again", "_n_item", "_n_of-for", "_v_down", "temp", "_v_qmodal", "year_range", "comp_less", "_n_i", "_a_than-from", "_v_off", "plus", "fraction", "interval", "interval_p_start", "interval_p_end", "_high_a_1",
                            "_and_c", "_v_into", "_v_from-to", "parenthetical", "_v_out", "_a_from", "_p_time", "_a_at", "_long_a_1", "_n_cur", "elliptical_n", "_n_2", "free_relative_q", "_x_prd", "_a_also", "numbered_hour", "_a_in", "comp_too", "reason"])
    if config['pretrained']:
        model = BartForConditionalGeneration.from_pretrained(model_type, config=model_config)
    else:
        model = BartForConditionalGeneration()
    model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

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

    trainer_args = Seq2SeqTrainingArguments(
        output_dir='models/' + config['name'],
        evaluation_strategy='epoch',
        learning_rate=config['learning_rate'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        weight_decay=config['weight_decay'],
        gradient_accumulation_steps=config['grad_accum_steps'],
        max_grad_norm=config['grad_norm'],
        warmup_steps=config['warmup_steps'],
        save_strategy='epoch',
        save_total_limit=1,
        num_train_epochs=config['max_epochs'],
        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        report_to='wandb' if config['wandb'] else 'none',
        no_cuda=True
        )

    trainer = Seq2SeqTrainer(
        model,
        trainer_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['dev'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    if args.checkpoint:
        trainer.train(resume_from_checkpoint=args.checkpoint)
    else:
        trainer.train()

    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.save_state()

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', type=Path, help='directory path to config', default='config/config.yaml')
    argparser.add_argument('-cp', '--checkpoint', help='directory path to checkpoint to resume from', default='')

    args = argparser.parse_args()

    with args.config.open() as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train(config, args)

if __name__ == '__main__':
    main()