
$MODEL_NAME=marathi_transfer_Hindi
python main.py \
    --model_name marathi_transfer_from_Hindi \
    --input_folder $DATA_DIR/  \
    --dev_path   $DATA_DIR/UD_Marathi-UFAL/udmap_mr_ufal-um-dev.conllu \
    --test_path    $DATA_DIR/UD_Marathi-UFAL/udmap_mr_ufal-um-covered-test.conllu \
    --multilingual \
    --eval_folder ../eval \
    --save_to_path ../saved_models/ \
    --model_arc char_birnn_attn \
    --langs hi_hdtb \
    --augVocablang mr_ufal \
    --lang mr_ufal \
    --test_conll \
    --gold_file $DATA_DIR/UD_Marathi-UFAL/mr_ufal-um-dev.conllu \
    --gold_test_file $DATA_DIR/UD_Marathi-UFAL/mr_ufal-um-dev.conllu \
    --lang_codes ../utils/lang_codes_updated.txt \
    --use_langid \
    --use_char_attention \
    --tot_epochs 100  2>&1 | tee ${MODEL_NAME}.log