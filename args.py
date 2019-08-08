def init_config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynet-mem", default=1000, type=int)
    parser.add_argument("--dynet-seed", default=5783287, type=int)
    parser.add_argument("--dynet-gpu")

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--eval_folder", type=str, default="../eval")
    parser.add_argument("--lang", default=None, help="the target language")
    parser.add_argument("--train_ensemble", default=False, action="store_true")
    parser.add_argument("--full_data_path", type=str, default=None, help="when train_ensemble is true, this one is the full data path from which to load vocabulary.")
    parser.add_argument("--train_path", default=None, type=str)
    # parser.add_argument("--train_path", default="../datasets/english/debug_train.bio", type=str)
    parser.add_argument("--monolingual_data_path", default=None, type=str)
    parser.add_argument("--dev_path", default="../datasets/english/eng.dev.bio.conll", type=str)
    parser.add_argument("--test_path", default="../datasets/english/eng.test.bio.conll", type=str)
    parser.add_argument("--new_test_path", default="../datasets/english/eng.test.bio.conll", type=str)
    parser.add_argument("--new_test_conll", default="../datasets/english/eng.test.bio.conll", type=str)
    parser.add_argument("--save_to_path", default="../saved_models/")
    parser.add_argument("--load_from_path", default=None)

    parser.add_argument("--cap_ratio_path", default=None, type=str)
    parser.add_argument("--non_ent_path", default=None, type=str)
    parser.add_argument("--min_edit_dist", default=-1, type=int)

    parser.add_argument("--train_filename_path", default=None, type=str)
    parser.add_argument("--dev_filename_path", default=None, type=str)
    parser.add_argument("--test_filename_path", default=None, type=str)
    parser.add_argument("--gold_pos_test_filename_path", default=None, type=str)

    # oromo specific argument
    # No matter orm_norm or orm_lower, the char representation is from the original word
    parser.add_argument("--lower_case_model_path", type=str, default=None)
    parser.add_argument("--train_lowercase_oromo", default=False, action="store_true")
    parser.add_argument("--oromo_normalize", default=False, action="store_true", help="if train lowercase model, not sure if norm also helps, this would loss a lot of information")

    parser.add_argument("--model_arc", default="char_cnn", choices=["char_cnn", "char_birnn", "char_birnn_cnn", "sep", "sep_cnn_only","char_birnn_attn"], type=str)
    parser.add_argument("--tag_emb_dim", default=50, type=int)
    parser.add_argument("--pos_emb_dim", default=64, type=int)
    parser.add_argument("--char_emb_dim", default=30, type=int)
    parser.add_argument("--word_emb_dim", default=100, type=int)
    parser.add_argument("--cnn_filter_size", default=30, type=int)
    parser.add_argument("--cnn_win_size", default=3, type=int)
    parser.add_argument("--rnn_type", default="lstm", choices=['lstm', 'gru'], type=str)
    parser.add_argument("--hidden_dim", default=200, type=int, help="token level rnn hidden dim")
    parser.add_argument("--char_hidden_dim", default=25, type=int, help="char level rnn hidden dim")
    parser.add_argument("--layer", default=1, type=int)
    parser.add_argument("--lm_obj", default=False, action="store_true")
    parser.add_argument("--lm_llama", default=False, action="store_true")
    parser.add_argument("--lm_param", default=0.5, type=float)

    parser.add_argument("--replace_unk_rate", default=0.0, type=float, help="uses when not all words in the test data is covered by the pretrained embedding")
    parser.add_argument("--remove_singleton", default=False, action="store_true")
    parser.add_argument("--map_pretrain", default=False, action="store_true")
    parser.add_argument("--map_dim", default=100, type=int)
    parser.add_argument("--pretrain_fix", default=False, action="store_true")

    parser.add_argument("--output_dropout_rate", default=0.5, type=float, help="dropout applied to the output of birnn before crf")
    parser.add_argument("--emb_dropout_rate", default=0.3, type=float, help="dropout applied to the input of token-level birnn")
    parser.add_argument("--valid_freq", default=500, type=int)
    parser.add_argument("--tot_epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--init_lr", default=0.015, type=float)
    parser.add_argument("--lr_decay", default=False, action="store_true")
    parser.add_argument("--decay_rate", default=0.05, action="store", type=float)
    parser.add_argument("--patience", default=3, type=int)

    parser.add_argument("--tagging_scheme", default="bio", choices=["bio", "bioes"], type=str)

    parser.add_argument("--data_aug", default=False, action="store_true", help="If use data_aug, the train_path should be the combined training file")
    parser.add_argument("--aug_lang", default="english", help="the language to augment the dataset")
    parser.add_argument("--aug_lang_train_path", default=None, type=str)
    parser.add_argument("--tgt_lang_train_path", default="../datasets/english/eng.train.bio.conll", type=str)

    parser.add_argument("--pretrain_emb_path", type=str, default=None)
    parser.add_argument("--res_discrete_feature", default=False, action="store_true", help="residual use of discrete features")

    parser.add_argument("--feature_birnn_hidden_dim", default=50, type=int, action="store")

    parser.add_argument("--use_discrete_features", default=False, action="store_true", help="David's indicator features")
    parser.add_argument("--split_hashtag", default=False, action="store_true", help="indicator of preceding hashtags")
    parser.add_argument("--cap", default=False, action="store_true", help="capitalization feature")
    parser.add_argument("--feature_dim", type=int, default=10, help="dimension of discrete features")

    parser.add_argument("--use_brown_cluster", default=False, action="store_true")
    parser.add_argument("--brown_cluster_path", action="store", type=str, help="path to the brown cluster features")
    parser.add_argument("--brown_cluster_num", default=0, type=int, action="store")
    parser.add_argument("--brown_cluster_dim", default=30, type=int, action="store")

    parser.add_argument("--use_gazatter", default=False, action="store_true")
    parser.add_argument("--use_morph", default=False, action="store_true")
    parser.add_argument("--use_pos", default=False, action="store_true")
    parser.add_argument("--pos_emb_file", default=None, type=str)
    parser.add_argument("--pos_train_file", default=None, type=str)
    parser.add_argument("--pos_dev_file", default=None, type=str)
    parser.add_argument("--pos_test_file", default=None, type=str)
    parser.add_argument("--gold_file", default=None, type=str)
    parser.add_argument("--gold_test_file", default=None, type=str)

    # CRF decoding
    parser.add_argument("--interp_crf_score", default=False, action="store_true", help="if True, interpolate between the transition and emission score.")
    # post process arguments
    parser.add_argument("--label_prop", default=False, action="store_true")
    parser.add_argument("--confidence_num", default=2, type=str)
    parser.add_argument("--author_file", default=None, type=str)
    parser.add_argument("--lookup_file", default=None, type=str)
    parser.add_argument("--freq_ngram", default=20, type=int)
    parser.add_argument("--stop_word_file", default=None, type=str)

    parser.add_argument("--isLr", default=False, action="store_true")
    parser.add_argument("--valid_on_full", default=False, action="store_true")
    parser.add_argument("--valid_using_split", default=False, action="store_true")
    parser.add_argument("--setEconll", type=str, default=None, help="path to the full setE conll file")
    parser.add_argument("--setEconll_10", type=str, default=None, help="path to the 10% setE conll file")
    parser.add_argument("--score_file", type=str, default=None,help="path to the scoring file for full setE conll file")
    parser.add_argument("--score_file_10", type=str, default=None, help="path to the scoring file for 10% setE conll file")

    parser.add_argument("--gold_setE_path", type=str, default="../ner_score/")
    # Use trained model to test
    parser.add_argument("--mode", default="train", type=str, choices=["train", "test_2", "test_1", "ensemble", "pred_ensemble","test_new"],
                        help="test_1: use one model; test_2: use lower case model and normal model to test oromo; "
                             "ensemble: CRF ensemble; pred_ensemble: ensemble prediction results")
    parser.add_argument("--ensemble_model_paths", type=str, help="each line in this file is the path to one model")

    # Partial CRF
    parser.add_argument("--use_partial", default=False, action="store_true")

    # Active Learning
    parser.add_argument("--ngram", default=2, type=int)
    parser.add_argument("--to_annotate", type=str,default="./annotate.txt")
    parser.add_argument("--entropy_threshold", type=float, default=None)
    parser.add_argument("--use_CFB", default=False, action="store_true")
    parser.add_argument("--sumType",default=False, action="store_true")
    parser.add_argument("--use_similar_label", default=False, action="store_true")
    parser.add_argument("--SPAN_wise", default=False, action="store_true", help="get span wise scores, even if there are duplicates.")
    parser.add_argument("--k", default=200, type=int, help="fixed number of spans to annotate")
    parser.add_argument("--clusters", default=400, type=int, help="fixed number of spans to annotate")
    parser.add_argument("--debug", type=str)
    parser.add_argument("--clusterDetails", type=str)
    parser.add_argument("--sqrtnorm", action="store_true", default=False)
    parser.add_argument("--activeLearning", action="store_true", default=False)
    parser.add_argument("--use_label_propagation", action="store_true", default=False)
    parser.add_argument("--label_prop_num", type=int, default=5)
    parser.add_argument("--cosineAL", action="store_true", default=False)
    parser.add_argument("--use_centroid", action="store_true", default=False)

    # Format of test output
    parser.add_argument("--test_conll", default=False, action="store_true")
    parser.add_argument("--fixedVocab", default=False, action="store_true", help="for loading pre-trained model")
    parser.add_argument("--fineTune", default=False, action="store_true", help="for loading pre-trained model")
    parser.add_argument("--run",default=0, type=int)
    parser.add_argument("--misc",default=False, action="store_true")

    # Task 
    parser.add_argument("--task",type=str, default="sigmorph")

    #Add multiple languages
    parser.add_argument("--input_folder", default="/Users/aditichaudhary/Documents/CMU/SIGMORPH/myNRF/data", type=str)
    parser.add_argument("--pos_folder", default="/Users/aditichaudhary/Documents/CMU/SIGMORPH/myNRF/data/POS_Folder", type=str)

    parser.add_argument("--lang_codes",
                        default="/Users/aditichaudhary/Documents/CMU/Lorelei/LORELEI_NER/utils/lang_codes.txt",
                        type=str)
    parser.add_argument("--langs", type=str, default="en/hi")
    parser.add_argument("--augVocablang", type=str, default=None)
    parser.add_argument("--use_langid", action="store_true", default=False)
    parser.add_argument("--use_token_langid", action="store_true", default=False)
    parser.add_argument("--use_char_attention", action="store_true", default=False)
    parser.add_argument("--use_lang_specific_decoder", action="store_true", default=False)
    parser.add_argument("--multilingual", default=False, action="store_true") #TO use data from from multiple languages, currently supported for sigmorph

    # Typological features
    parser.add_argument("--typology_features_file", type=str, default=None)
    parser.add_argument("--use_typology", default=False, action="store_true")
    parser.add_argument("--typology_feature_dim", default=10, type=int)

    #Lang-Gender Specific
    parser.add_argument("--use_gender_specific", action="store_true", default=False)
    parser.add_argument("--no_gender", action="store_true", default=False)
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--sent_count", type=int, default=10000000)
    parser.add_argument("--myNRF", action='store_true', default=False)

    #Use only character-based representations
    parser.add_argument("--only_char", action="store_true", default=True)

    args = parser.parse_args()

    # We are not using uuid to make a unique time stamp, since I thought there is no need to do so when we specify a good model_name.

    # If use score_10pct.sh, put the setE_10pct.txt as the dev_path
    # If use valid_using_split, set the test_path and setEconll to be the splitted version, this is used for full setE testing
    if args.train_ensemble:
        # model_name = ens_1_ + original
        # set dynet seed manually
        ens_no = int(args.model_name.split("_")[1])
        # dyparams = dy.DynetParams()
        # dyparams.set_random_seed(ens_no + 5783287)
        # dyparams.init()

        import dynet_config
        dynet_config.set(random_seed=ens_no + 5783290)
        # if args.cuda:
        #     dynet_config.set_gpu()

        # args.train_path = args.train_path.split(".")[0] + "_" + str(ens_no) + ".conll"

    if args.full_data_path is None:
        args.full_data_path = args.train_path
    args.save_to_path = args.save_to_path + args.model_name + ".model"
#    args.gold_setE_path = args.gold_setE_path + args.lang + "_setE_edl.tac"
    print(args)
    return args
