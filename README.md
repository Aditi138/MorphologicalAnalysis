# Morphological Analysis - System submission for SIGMORPHON 2019 Task 2.


``` $DATA_DIR = data directory containing the treebanks obtained from [[here]](https://github.com/sigmorphon/2019)```

1. Data Pre-processing to convert UniMorph data into key, value pairs.
Original format: ```4	होते	असणे	_	_	PST;3;MASC;FIN;V;PL	_	_	_	_```

``` python helper_scripts/pre_process.py
        --input $DATA_DIR/UD_Marathi-UFAL \
        --attributes helper_scripts/attributes.txt \
        --train $DATA_DIR/UD_Marathi-UFAL/mr_ufal_um_train.conllu \
        --dev $DATA_DIR/UD_Marathi-UFAL/mr_ufal_um_dev.conllu \
        --test $DATA_DIR/UD_Marathi-UFAL/mr_ufal_um_test.conllu
```

This will create the following files in the ```$DATA_DIR/UD_Marathi-UFAL``` namely:
    ```udmap_pos_mr_ufal_um_train.conllu
       udmap_pos_mr_ufal_um_dev.conllu
       udmap_pos_mr_ufal_um_test.conllu```

       in the following format
       ``` 4	होते	असणे	 V	_	POS=V|Tense=PST|Person=3|Gender=MASC|Finiteness=FIN|Number=PL	_	_	_	_
        ```
A set of files with POS in the 4th coloumn and remaining tags in the 6th coloumn in the following format.
    ```pretrain_udmap_pos_mr_ufal_um_train.conllu
       pretrain_udmap_pos_mr_ufal_um_dev.conllu
       pretrain_udmap_pos_mr_ufal_um_test.conllu
        4	होते	असणे	 V	_	Tense=PST|Person=3|Gender=MASC|Finiteness=FIN|Number=PL	_	_	_	_```
       Additionally, it will create the following files for training an independent POS tagger which are in the conll format:
       ```
       POS_train.conllu
       POS_dev.conllu
       POS_test.conllu ```
       
       ```होते	V ```


2. To train the MDCRF model, where we make independent predictions for each feature (POS, Gender, etc) using a hierarchical-neural CRF model.
 ``` cd MorphologicalAnalysis/commands 
    python main.py \
    --model_name marathi_transfer_from_Hindi \
    --input_folder $DATA_DIR/  \
    --dev_path   $DATA_DIR/UD_Hindi-HDTB/udmap_hi_hdtb-um-dev.conllu \
    --test_path    $DATA_DIR/UD_Hindi-HDTB/udmap_hi_hdtb-um-covered-test.conllu \
    --multilingual \
    --eval_folder ../eval \
    --save_to_path ../saved_models/ \
    --model_arc char_birnn_attn \
    --langs hi_hdtb \
    --augVocablang mr_ufal/sa_ufal \
    --lang hi_hdtb \
    --test_conll \
    --gold_file $DATA_DIR/UD_Hindi-HDTB/hi_hdtb-um-dev.conllu \
    --gold_test_file $DATA_DIR/UD_Hindi-HDTB/hi_hdtb-um-test.conllu \
    --lang_codes ../utils/lang_codes_updated.txt \
    --use_langid \
    --use_char_attention \
    --tot_epochs 100  ```


    Include language codes which are used during training the model in ```--langs ```. 
If you want a zero-shot transfer, you need to include those target languages in ```--augVocablang```.
For instance, in the above case, Hindi is used for training a zero-shot transfer model to be used later for Marathi and Sanskrit.
    For testing on a language, add the following two arguments and change the test path and re-run the above command.
    ``` --mode test_1 \
        --load_from_path ../saved_models/marathi_transfer_from_Hindi.model \
        --test_path    $DATA_DIR/UD_Marathi-UFAL/udmap_mr_ufal-um-covered-test.conllu \
        --gold_test_file $DATA_DIR/UD_Marathi-UFAL/mr_ufal-um-dev.conllu
     ```

    If you want to fineTune a transferred model for a target language say Marathi, run the below as a second step. This will load the above model stored in ```../saved_models``` and fine-tune over the target language data.
    ``` python main.py \
    --model_name marathi_transfer_from_Hindi_fineTuned \
    --input_folder $DATA_DIR/  \
    --train_path $DATA_DIR/UD_Marathi-UFAL/udmap_mr_ufal-um-train.conllu \
    --dev_path   $DATA_DIR/UD_Marathi-UFAL/udmap_mr_ufal-um-dev.conllu \
    --test_path    $DATA_DIR/UD_Marathi-UFAL/udmap_mr_ufal-um-covered-test.conllu \
    --multilingual \
    --fineTune \
    --load_from_path ../saved_models/marathi_transfer_from_Hindi.model \
    --eval_folder ../eval \
    --save_to_path ../saved_models/ \
    --model_arc char_birnn_attn \
    --langs hi_hdtb \
    --augVocablang mr_ufal/sa_ufal \
    --lang mr_ufal \
    --test_conll \
    --gold_file $DATA_DIR/UD_Marathi-UFAL/mr_ufal-um-dev.conllu \
    --gold_test_file $DATA_DIR/UD_Marathi-UFAL/mr_ufal-um-dev.conllu \
    --lang_codes ../utils/lang_codes_updated.txt \
    --use_langid \
    --use_char_attention \
    --tot_epochs 100  ```

