__author__ = 'chuntingzhou and aditichaudhary'
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import evaluate_2019_task2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



def evaluate(data_loader, path, model, model_name, pos_tags=None, type="dev",task="sigmorph"):

    sents, char_sents, tgt_tags, discrete_features, bc_feats, _ = data_loader.get_data_set_morph(path, args.lang,
                                                                                               source="dev")

    prefix = model_name + "_" + str(uid)
    predictions = []
    sentences = []
    i = 0
    gold_file = args.gold_file
    score_sent = {}
    num = 0
    langid = data_loader.word_to_id["<" + args.lang + ">"]

    if args.use_pos:
        #Running the encoder to get similarity scores before
        if type == "test" and args.activeLearning:
            model.word_embedding_weights = []
            model.token_embeddings = []
            model.tokens = []
            for sent, char_sent, discrete_feature, bc_feat, b_pos in zip(sents, char_sents, discrete_features,
                                                                         bc_feats, pos_tags):
                dy.renew_cg()
                sent, char_sent, discrete_feature, bc_feat, b_pos, b_lang = [sent], [char_sent], [discrete_feature], [
                    bc_feat], [b_pos], [[langid for _ in range(len(sent))]]
                model.getscores(sent, char_sent, discrete_feature, bc_feat, b_pos=b_pos,
                                                     langs=b_lang, training=False, type=type)


            if args.cosineAL:
                model.constructWordSimilarity()

            else:
                model.getSimilarityMatrix(tgt_tags=tgt_tags)
        for sent, char_sent, discrete_feature, bc_feat, b_pos in zip(sents, char_sents, discrete_features,
                                                                       bc_feats, pos_tags):
            dy.renew_cg()
            sent, char_sent, discrete_feature, bc_feat, b_pos, b_lang = [sent], [char_sent], [discrete_feature], [bc_feat], [b_pos], [ [langid for _ in range(len(sent))]]
            best_scores, best_paths = model.eval(sent, char_sent, discrete_feature, bc_feat,b_pos=b_pos, langs=b_lang, training=False, type=type)
            one_prediction = [[] for _ in range(len(sent[0]))]
            one_gold = [[] for _ in range(len(sent[0]))]
            for j in range(len(sent[0])):
                one_prediction[j].append(data_loader.pos_idx2label[b_pos[0][j]])

            sent_score = 0.0
            for key, gold_tgt_tags in tgt_tags.items():
                if key == "Gender" and args.no_gender:
                    continue
                best_path = best_paths[key]
                sent_score += best_scores[key]
                #print(len(best_path), len(gold_tgt_tags[i]))
                assert len(best_path) == len(gold_tgt_tags[i])
                for token_num, pred_tag_feat in enumerate(best_path):
                    if data_loader.id2tags[key][pred_tag_feat] == "_":
                        continue
                    tag_name = data_loader.id2tags[key][pred_tag_feat]
                    if args.use_gender_specific:
                        tag_name = to_replace.get(tag_name, tag_name)
                    one_prediction[token_num].append(tag_name)
            sent_score = sent_score  / len(best_scores)
            sentence_key =  " ".join([str(x) for x in sent[0]])
            score_sent[sentence_key] = sent_score
            # acc = model.crf_decoder.cal_accuracy(best_path, tgt_tag)
            # tot_acc += acc
            predictions.append(one_prediction)
            sentences.append(sent)
            i += 1
            if i % 1000 == 0:
                print("Testing processed %d lines " % i)
    else:
        # Running the encoder to get similarity scores before
        if type == "test" and args.activeLearning:
            model.word_embedding_weights = []
            model.token_embeddings = []
            model.tokens = []
            if args.use_similar_label:
                for sent, char_sent, discrete_feature, bc_feat in zip(sents, char_sents, discrete_features,
                                                                         bc_feats):
                    dy.renew_cg()
                    sent, char_sent, discrete_feature, bc_feat, b_lang = [sent], [char_sent], [discrete_feature], [bc_feat], [[langid for _ in range(len(sent))]]
                    model.getscores(sent, char_sent, discrete_feature, bc_feat,langs=b_lang, training=False, type=type)
            if args.use_similar_label:
                model.getSimilarityMatrix(tgt_tags=tgt_tags)

        for sent, char_sent, discrete_feature, bc_feat in zip(sents, char_sents, discrete_features, bc_feats):
            dy.renew_cg()
            sent, char_sent, discrete_feature, bc_feat, b_lang = [sent], [char_sent], [discrete_feature], [bc_feat], [ [langid for _ in range(len(sent))]]
            best_scores, best_paths = model.eval(sent, char_sent, discrete_feature, bc_feat, langs=b_lang, training=False,type=type)
            one_prediction = [[] for _ in range(len(sent[0]))]
            sent_score = 0.0
            for key, gold_tgt_tags in tgt_tags.items():
                best_path = best_paths[key]
                sent_score += best_scores[key]
                #print(len(best_path), len(gold_tgt_tags[i]))
                assert len(best_path) == len(gold_tgt_tags[i])
                for token_num, pred_tag_feat in enumerate(best_path):
                    if data_loader.id2tags[key][pred_tag_feat] == "_":
                        continue
                    tag_name = data_loader.id2tags[key][pred_tag_feat]
                    if args.use_gender_specific and key == "Gender":
                        tag_name = to_replace.get(tag_name, tag_name)
                    one_prediction[token_num].append(tag_name)
            # acc = model.crf_decoder.cal_accuracy(best_path, tgt_tag)
            # tot_acc += acc
            sent_score = sent_score / len(best_scores)
            sentence_key = " ".join([str(x) for x in sent[0]])
            score_sent[sentence_key] = sent_score
            predictions.append(one_prediction)
            sentences.append(sent)

            i += 1
            if i % 1000 == 0:
                print("Testing processed %d lines " % i)

        num +=1

    if type == "dev" and "dev" in path:
        pred_output_fname = path.split("udmap")[0] + "/sigmorph_dev_pred.conllu"
    elif type == "test":
        pred_output_fname = path.split("udmap")[0] + "/sigmorph_test_pred.conllu"
        gold_file = args.gold_test_file
    else:
        pred_output_fname = "%s/%s_pred_output.txt" % (args.eval_folder,prefix)

    with open(pred_output_fname, "w") as fout:
        for sent, pred in zip(sentences, predictions):
            info = ["_" for _ in range(10)]
            index = 1
            if args.use_langid:
                sent = sent[0][1:-1]
                pred = pred[1:-1]
            else:
                sent = sent[0]
            for s, p in zip(sent, pred):
                new_p = []
                for fe in p:
                    if fe != "_":  
                        new_p.append(fe)
                p = new_p
                if len(p) == 0:
                    p = "_"
                elif len(p) > 1:
                    p = ";".join(p)
                else:
                    p = p[0]
                info[0], info[1],info[5]  = str(index), data_loader.id_to_word[int(s)] , p
                fout.write("\t".join(info) + "\n")
                index += 1
                #fout.write(data_loader.id_to_word[int(s)] + " " + data_loader.id_to_tag[g] + " " + data_loader.id_to_tag[p] + "\n")
            fout.write("\n")

    acc, f1,  precision, recall, prediction_pairs, gold_pairs = evaluate_2019_task2.main(out_file=pred_output_fname, ref_file=gold_file)
    if type == "dev":
        os.system("rm %s" % (pred_output_fname,))
    return acc, precision, recall, f1, prediction_pairs, gold_pairs, score_sent

def replace_singletons(data_loader, sents, replace_rate):
    new_batch_sents = []
    for sent in sents:
        new_sent = []
        for word in sent:
            if word in data_loader.singleton_words:
                new_sent.append(word if np.random.uniform(0., 1.) > replace_rate else data_loader.word_to_id["<unk>"])
            else:
                new_sent.append(word)
        new_batch_sents.append(new_sent)
    return new_batch_sents


def main(args):
    prefix = args.model_name + "_" + str(uid)
    print("PREFIX: %s" % prefix)
    ner_data_loader = NER_DataLoader(args)
    train_pos_tags_idx, dev_pos_tags_idx, test_pos_tags_idx = [], [], []


    #Loading training data from multiple languages, specified in args.langs argument
    #A language id as added for each data point
    if args.multilingual:
        sents, char_sents, tgt_tags, discrete_features, bc_features, known_tags, langs, typological_features = [], [], defaultdict(list), [], [], defaultdict(list), [], []
        train_pos_tags, dev_pos_tags, test_pos_tags = [], [], []
        pos_tagset = set()
        all_langs = args.langs.split("/")
        print(all_langs)
        for lang in all_langs:
            input_folder = args.input_folder + "/" + "UD_" + ner_data_loader.code_to_lang[lang]  + "//"
            print("Reading files from folder", input_folder)
            train_path = None
            if args.train_path is not None:
                train_path = args.train_path
            else:
                for [path, dir, files] in os.walk(input_folder):
                    for file in files:
                        if file.startswith(ner_data_loader.filestart) and file.endswith("train.conllu"):
                            train_path = input_folder + file
                            break
                    break
            if not os.path.exists(train_path):
                print("Train Feature file not exists", train_path)
                continue
            print("Reading from, ", train_path)
            lang_sents, lang_char_sents, lang_tgt_tags, lang_discrete_features, lang_bc_features, lang_known_tags = ner_data_loader.get_data_set_morph(
                train_path, lang)

            sents += lang_sents
            char_sents += lang_char_sents
            for key, lang_tags in lang_tgt_tags.items():
                tgt_tags[key] += lang_tags
                known_tags[key] += lang_known_tags[key]

            #TODO these two features are unused, need to remove them
            discrete_features += lang_discrete_features
            bc_features += lang_bc_features
            langs += ["<"+lang+">" for _ in range(len(lang_sents))]

            #For training the model with MDCRF+POS, where POS embeddings are concatenated with token embeddings
            #For this a separate POS tagger is used and the predictions are used  below.
            if args.use_pos:
                pos_folder = args.pos_folder  + "/" + "UD_" + ner_data_loader.code_to_lang[lang]  + "//"
                pos_train_file = pos_folder + "POS_train.conll"
                print("Reading from, ", pos_train_file)
                lang_train_pos_tags = ner_data_loader.get_pos_data_set(pos_train_file, pos_tagset)
                train_pos_tags += lang_train_pos_tags
                print(len(lang_sents), len(lang_train_pos_tags))
                assert len(lang_sents) == len(lang_train_pos_tags)

                if lang == args.lang:
                    pos_dev_file = pos_folder + "POS_pred_dev.conll"
                    print("Reading POS dev file", pos_dev_file)
                    lang_dev_pos_tags = ner_data_loader.get_pos_data_set(pos_dev_file, pos_tagset, isDev=True)
                    dev_pos_tags += lang_dev_pos_tags
                    pos_test_file = pos_folder + "POS_test.conll"
                    lang_test_pos_tags = ner_data_loader.get_pos_data_set(pos_test_file, pos_tagset, isDev=True)
                    test_pos_tags += lang_test_pos_tags

            # For loading typological features per language
            if args.use_typology:
                typological_feature = ner_data_loader.pre_computed_features[lang]
                typological_features += [typological_feature for _ in range(len(lang_sents))]


        #When fine-tuning on just  a target language.
        if args.fineTune:
            input_folder = args.input_folder + "/" + "UD_" + ner_data_loader.code_to_lang[args.lang] + "//"
            if args.train_path is not None:
                train_path = args.train_path
            else:
                for [path, dir, files] in os.walk(input_folder):
                    for file in files:
                        if file.startswith(ner_data_loader.filestart) and file.endswith("train.conllu"):
                            train_path = input_folder + file
                            break
                    break
            
            print("FineTune: Reading from, ", train_path)
            lang_sents, lang_char_sents, lang_tgt_tags, lang_discrete_features, lang_bc_features, lang_known_tags = ner_data_loader.get_data_set_morph(
                train_path, args.lang)

            sents = lang_sents[:args.sent_count]
            char_sents = lang_char_sents[:args.sent_count]
            for key, lang_tags in lang_tgt_tags.items():
                tgt_tags[key] = lang_tags[:args.sent_count]
                known_tags[key] = lang_known_tags[key][:args.sent_count]
            discrete_features = lang_discrete_features[:args.sent_count]
            bc_features = lang_bc_features[:args.sent_count]
            langs = ["<"+args.lang+">" for _ in range(len(sents))]
            if args.use_pos:
                pos_folder = args.pos_folder + "/" + "UD_" + ner_data_loader.code_to_lang[args.lang] + "//"
                print(args.lang)
                pos_train_file = pos_folder + "POS_train.conll"
                print("Reading from, ", pos_train_file)
                lang_train_pos_tags = ner_data_loader.get_pos_data_set(pos_train_file, pos_tagset)[:args.sent_count]
                train_pos_tags = lang_train_pos_tags
                print(len(sents), len(lang_train_pos_tags))
                assert len(sents) == len(lang_train_pos_tags)

                pos_dev_file = pos_folder + "POS_pred_dev.conll"
                print("Reading POS dev file", pos_dev_file)
                lang_dev_pos_tags = ner_data_loader.get_pos_data_set(pos_dev_file, pos_tagset, isDev=True)
                dev_pos_tags += lang_dev_pos_tags
                pos_test_file = pos_folder + "POS_test.conll"
                lang_test_pos_tags = ner_data_loader.get_pos_data_set(pos_test_file, pos_tagset, isDev=True)
                test_pos_tags += lang_test_pos_tags

        if args.use_pos:
            pos_tagset.add("_")
            print(pos_tagset)
            ner_data_loader.pos_labe2idx = ner_data_loader.get_vocab_from_set(pos_tagset)
            ner_data_loader.pos_vocab_size = len(ner_data_loader.pos_labe2idx.keys())
            ner_data_loader.pos_idx2label = {v: k for k, v in ner_data_loader.pos_labe2idx.items()}
            ner_data_loader.pretrain_pos_emb = loadPretrainedEmbedding(args.pos_emb_file, args.pos_emb_dim,
                                                                       ner_data_loader.pos_labe2idx)

            for index, data_point in enumerate(train_pos_tags):
                train_pos_tags_idx.append([ner_data_loader.pos_labe2idx[token] for token in data_point])
            for data_point in dev_pos_tags:
                dev_pos_tags_idx.append([ner_data_loader.pos_labe2idx[token] for token in data_point])
            for data_point in test_pos_tags:
                test_pos_tags_idx.append([ner_data_loader.pos_labe2idx[token] for token in data_point])

    elif not args.data_aug:
        sents, char_sents, tgt_tags, discrete_features, bc_features, known_tags = ner_data_loader.get_data_set(
            args.train_path, args.lang)
    else:
        sents_tgt, char_sents_tgt, tags_tgt, dfs_tgt, bc_feats_tgt, known_tags_tgt = ner_data_loader.get_data_set(
            args.tgt_lang_train_path, args.lang)
        sents_aug, char_sents_aug, tags_aug, dfs_aug, bc_feats_aug, known_tags_aug = ner_data_loader.get_data_set(
            args.aug_lang_train_path, args.aug_lang)
        sents, char_sents, tgt_tags, discrete_features, bc_features, known_tags = sents_tgt + sents_aug, char_sents_tgt + char_sents_aug, tags_tgt + tags_aug, dfs_tgt + dfs_aug, bc_feats_tgt + bc_feats_aug, known_tags_tgt + known_tags_aug



    print("Data set size (train): %d" % len(sents))
    print("Number of discrete features: ", ner_data_loader.num_feats)
    epoch = bad_counter = updates = tot_example = cum_loss = 0
    patience = args.patience
    print("Dev POS len", len(dev_pos_tags_idx))
    display_freq = 100
    valid_freq = args.valid_freq
    batch_size = args.batch_size
    print("Using Char Birnn Attn model!")
    model = BiRNN_ATTN_CRF_model(args, ner_data_loader, None)
    inital_lr = args.init_lr

    if args.fineTune:
        print("Loading pre-trained model!")
        model.load()

    trainer = dy.MomentumSGDTrainer(model.model, inital_lr, 0.9)

    def _check_batch_token(batch, id_to_vocab, fout):
        for line in batch:
            fout.write(" ".join([id_to_vocab[i] for i in line]) + "\n")

    def _check_batch_tags(batch, id2tags, fout):
        for feat, sent_tags in batch.items():
            fout.write("Printing tags for feature: {0}".format( feat))
            for line in sent_tags:
                fout.write(" ".join([id2tags[feat][i] for i in line]) + "\n")

    def _check_batch_char(batch, id_to_vocab):
        for line in batch:
            print([u" ".join([id_to_vocab[c] for c in w]) for w in line])

    lr_decay = args.decay_rate


    valid_history = []
    best_results = [0.0, 0.0, 0.0, 0.0]
    sent_index = [i for i in range(len(sents))]


    #Starting training
    while epoch <= args.tot_epochs:
        if args.use_pos:
            batches = make_bucket_batches(
                zip(sents, char_sents, discrete_features, bc_features, langs, train_pos_tags_idx, sent_index,), tgt_tags, known_tags, batch_size)
            for b_sents, b_char_sents, b_feats, b_bc_feats,b_langs,b_pos_tags, _, b_tgt_tags,b_known_tags in batches:

                dy.renew_cg()

                if args.replace_unk_rate > 0.0:
                    b_sents = replace_singletons(ner_data_loader, b_sents, args.replace_unk_rate)

                token_size = len(b_sents[0])
                lang_batch = []
                for _ in range(len(b_sents)):
                    lang_batch.append([ner_data_loader.word_to_id[b_langs[0]] for _ in range(token_size)])
                loss = model.cal_loss(b_sents, b_char_sents, b_tgt_tags, b_feats, b_bc_feats, b_known_tags, pos_tags=b_pos_tags,langs=lang_batch, training=True)

                loss_val = loss.value()
                cum_loss += loss_val * len(b_sents)
                tot_example += len(b_sents)

                updates += 1
                loss.backward()
                trainer.update()

                if updates % display_freq == 0:
                    print("Epoch = %d, Updates = %d, CRF Loss=%f, Accumulative Loss=%f." % (epoch, updates, loss_val, cum_loss*1.0/tot_example))
                if updates % valid_freq == 0:
                    acc, precision, recall, f1, _, _ = evaluate(ner_data_loader, args.dev_path, model,
                                                                       args.model_name, pos_tags=dev_pos_tags_idx)

                    print(acc, f1)
                    if len(valid_history) == 0 or f1 > max(valid_history):
                        bad_counter = 0
                        best_results = [acc, precision, recall, f1]
                        if updates > 0:
                            print("Saving the best model so far.......", model.save_to)
                            model.save()
                    else:
                        bad_counter += 1
                        if args.lr_decay and bad_counter >= 3 and os.path.exists(args.save_to_path):
                            bad_counter = 0
                            model.load()
                            lr = inital_lr / (1 + epoch * lr_decay)
                            print("Epoch = %d, Learning Rate = %f." % (epoch, lr))
                            trainer = dy.MomentumSGDTrainer(model.model, lr)

                    if bad_counter > patience:
                        print("Early stop!")
                        print("Best on validation: acc=%f, prec=%f, recall=%f, f1=%f" % tuple(best_results))
                        if args.test_conll:
                            model.load_from = args.save_to_path
                            print("Loading best model from", model.load_from)
                            model.load()
                            acc, precision, recall, f1 = finalEval(args, dev_pos_tags_idx, f1, model, ner_data_loader, test_pos_tags_idx)

                            exit(0)
                    valid_history.append(f1)
        else:
            batches = make_bucket_batches(
                zip(sents, char_sents, discrete_features, bc_features, langs, sent_index), tgt_tags,known_tags, batch_size)
            for b_sents, b_char_sents, b_feats, b_bc_feats,b_langs, _, b_tgt_tags,b_known_tags in batches:
                dy.renew_cg()

                if args.replace_unk_rate > 0.0:
                    b_sents = replace_singletons(ner_data_loader, b_sents, args.replace_unk_rate)

                token_size = len(b_sents[0])
                lang_batch = []
                for _ in range(len(b_sents)):
                    lang_batch.append([ner_data_loader.word_to_id[b_langs[0]] for _ in range(token_size)])

                lm_batch = None
                loss = model.cal_loss(b_sents, b_char_sents, b_tgt_tags, b_feats, b_bc_feats, b_known_tags, langs=lang_batch, lm_batch=lm_batch,
                                      training=True)
                loss_val = loss.value()
                cum_loss += loss_val * len(b_sents)
                tot_example += len(b_sents)

                updates += 1
                loss.backward()
                trainer.update()

                if updates % display_freq == 0:
                    print("Epoch = %d, Updates = %d, CRF Loss=%f, Accumulative Loss=%f." % (
                    epoch, updates, loss_val, cum_loss * 1.0 / tot_example))
                if updates % valid_freq == 0:
                    acc, precision, recall, f1, _, _,_ = evaluate(ner_data_loader, args.dev_path, model, args.model_name, task=args.task)
                    print(acc, f1)

                    if len(valid_history) == 0 or f1 > max(valid_history):
                        bad_counter = 0
                        best_results = [acc, precision, recall, f1]
                        if updates > 0:
                            print("Saving the best model so far.......", model.save_to)
                            model.save()
                    else:
                        bad_counter += 1
                        if args.lr_decay and bad_counter >= 3 and os.path.exists(args.save_to_path):
                            bad_counter = 0
                            model.load()
                            lr = inital_lr / (1 + epoch * lr_decay)
                            print("Epoch = %d, Learning Rate = %f." % (epoch, lr))
                            trainer = dy.MomentumSGDTrainer(model.model, lr)

                    if bad_counter > patience:
                        print("Early stop!")
                        print("Best on validation: acc=%f, prec=%f, recall=%f, f1=%f" % tuple(best_results))
                        if args.test_conll:
                            model.load_from = args.save_to_path
                            print("Loading best model from", model.load_from)
                            model.load()
                            acc, precision, recall, f1 = finalEval(args, dev_pos_tags_idx, f1, model, ner_data_loader, test_pos_tags_idx)
                        exit(0)
                    valid_history.append(f1)

        epoch += 1

    if args.test_conll:
        model.load_from = args.save_to_path
        print("Loading best model from", model.load_from)
        model.load()
        acc, precision, recall, f1 = finalEval(args, dev_pos_tags_idx, f1, model, ner_data_loader, test_pos_tags_idx)

    print("All Epochs done.")


def finalEval(args, dev_pos_tags_idx, f1, model, ner_data_loader, test_pos_tags_idx):

    acc, precision, recall, f1, _, _,_ = evaluate(ner_data_loader, args.dev_path, model,
                                                args.model_name,
                                                pos_tags=dev_pos_tags_idx,
                                                type="dev",
                                                task=args.task)
    print("Dev: Acc: {0} Prec: {1} Recall: {2} F1: {3}".format(acc, precision, recall, f1))
    model.word_embedding_weights = []
    testacc, testprecision, testrecall, testf1,  gold_dict, output_dict, score_sentence = evaluate(ner_data_loader,args.test_path,
                                                                model,
                                                                args.model_name,
                                                                pos_tags=test_pos_tags_idx,
                                                                type="test",
                                                                task=args.task)
    print("Test: Acc: {0} Prec: {1} Recall: {2} F1: {3}".format(testacc, testprecision, testrecall, testf1))
    if args.activeLearning:
        createAnnotationOutput_SPAN_wise(args, model, ner_data_loader,  gold_dict, output_dict, score_sentence)
    return acc, precision, recall, f1


def createAnnotationOutput_SPAN_wise(args, model, data_loader, gold_dict, output_dict, score_sentence):
    # normalize all the entropy_spans ONLY DONE for the CFB
    sentence_index= []
    reverse = True
    if args.use_CFB:
        reverse = False

    # Order the sentences by entropy of the spans
    fout=  codecs.open(args.to_annotate, "w", encoding='utf-8')
    if args.sqrtnorm: 
        #Divide by count of the tokens
        for span, info in model.most_uncertain_entropy_spans.items():
            span_entropy = info[0] * 1.0 / math.sqrt(model.token_count[span])
            model.most_uncertain_entropy_spans[span] = (span_entropy, info[1], info[2], info[3], info[4], info[5], info[6])
    
    sorted_spans = sorted(model.most_uncertain_entropy_spans,  key=lambda k:model.most_uncertain_entropy_spans[k],reverse=reverse)
    print("Total unique spans: {0}".format(len(sorted_spans)))
    count_span = args.k
    count_tokens = args.k
	
    #DEBUG Print Span Entropy
    fdebug = codecs.open(args.debug, "w", encoding='utf-8')

    #Accumulate tokens by sentence
    sentence_index_sent = {}
    sentence_index_tokens = defaultdict(list)
    spans_with_tokens_less_than_5 = 0
    for sorted_span in sorted_spans:
        #Debug
        span_words= []
        #for t in sorted_span.split():
        #    span_words.append(data_loader.id_to_word[t])
        #f_debug.write(" ".join(span_words) + "\n")
        # if count_span <=0:
        #     break
        if count_tokens <=0:
            break
        (span_entropy, sentence_key, start, end,best_path, instance_entropy, sent_index, token_index) = model.most_uncertain_entropy_spans[sorted_span]
        sent = sentence_key.split()
        sent = [data_loader.id_to_word[int(token)] for token in sent]
        gold_path = gold_dict[" ".join(sent)]


        if args.use_similar_label:
            cluster_label = model.SimilarLabels[token_index]
            all_tokens_within_the_cluster = model.silhouette_vals[model.SimilarLabels == cluster_label]
            token_indices = model.token_indices[model.SimilarLabels == cluster_label]
            sorted_tokens_by_coeff, sorted_tokens_indices = zip(*(sorted(zip(all_tokens_within_the_cluster, token_indices), reverse=True)))
            top_most_token = data_loader.id_to_word[model.tokens[sorted_tokens_indices[0]]]
            prev_span_entropy = model.token_index_entropy_map[token_index]

            if args.use_centroid: #Use the centroid as the thing to annotate
                span_words = [top_most_token]
                token_index = sorted_tokens_indices[0]
                sent_index = model.token_index_sent_index_map[token_index]
                sentence_index.append(sent_index)
                sent = model.sent_index_sent_map[sent_index]
                sent = [data_loader.id_to_word[int(token)] for token in sent]
                gold_path = gold_dict[" ".join(sent)]
                sentence_index_sent[sent_index] = (sent, gold_path, output_dict[" ".join(sent)])
                sentence_index_tokens[sent_index].append(model.token_index_relative_idx_map[token_index])
                centroid_span_entropy = model.token_index_entropy_map[token_index]
            else:
                sentence_index.append(sent_index)
                sentence_index_sent[sent_index] = (sent, gold_path, output_dict[" ".join(sent)])
                sentence_index_tokens[sent_index].append(start)
                span_words.append(sent[start])
                centroid_span_entropy = model.token_index_entropy_map[token_index]

            gold_cluster_labels = model.gold_labels_per_cluster[cluster_label]
            cluster_items = model.mostSimilar[cluster_label]
            string_items = [data_loader.id_to_word[t] for t in cluster_items]
            fdebug.write(" ".join(span_words) + "\t" + str(span_entropy) + " / " + str(prev_span_entropy) + " / " + str(centroid_span_entropy) + "\t" + str(model.silhouette_vals[token_index])  +
                        "\t" +  top_most_token + "\t"  + str(sorted_tokens_by_coeff[0]) +   "\t" + ";".join(string_items) +  "\t" + "|".join(gold_cluster_labels) + "\n")
        else:
            sentence_index.append(sent_index)
            sentence_index_sent[sent_index] = (sent, gold_path, output_dict[" ".join(sent)])
            sentence_index_tokens[sent_index].append(start)
            span_words.append(sent[start])
            fdebug.write(" ".join(span_words) + "\t" + str(span_entropy) + "\n")

        if args.use_label_propagation:
            #Get all occcurrences of the token and sort them
            tokenEntropy = model.typeHeap[sorted_span]
            tokenInfo = model.heapInfo[sorted_span]
            sorted_tokenEntropy, sorted_tokenInfo = zip(*(sorted(zip(tokenEntropy, tokenInfo))))
            top_tokens_per_type = sorted_tokenInfo[len(sorted_tokenInfo)-args.label_prop_num:]
            if len(top_tokens_per_type) < 5:
                spans_with_tokens_less_than_5 +=1
                #print(len(top_tokens_per_type))
            for token_instance in top_tokens_per_type:
                (t_sentence_key, t_start, t_sent_index) = token_instance
                t_sent = [data_loader.id_to_word[int(token)] for token in t_sentence_key.split()]
                t_gold_path = gold_dict[" ".join(t_sent)]
                sentence_index.append(t_sent_index)
                sentence_index_sent[t_sent_index] = (t_sent, t_gold_path, output_dict[" ".join(t_sent)])
                sentence_index_tokens[t_sent_index].append(t_start)


        count_span -= 1
        count_tokens -= 1
        #print(spans_with_tokens_less_than_5)
        #One token per sentence to be annotated
        # for token, tag_label, gold_tag in zip(sent, path, gold_path):
        #     fout.write(token + "\t" + tag_label + "\t" + gold_tag + "\n")
        #
        # fout.write("\n")


    covered = set()
    count = 0
    for sent_index in sentence_index:
        if sent_index not in covered:
            covered.add(sent_index)
            (sent, gold_path, path) = sentence_index_sent[sent_index]
            for token_index in sentence_index_tokens[sent_index]:
                path[token_index] = "UNK"

            for token, tag_label, gold_tag in zip(sent, path, gold_path):
                fout.write(token + "\t" + tag_label + "\t" + gold_tag + "\n")
                if tag_label == "UNK":
                    count += 1

            fout.write("\n")


    print("Total unique spans for exercise: {0}".format(args.k))
    if args.use_label_propagation:
        print("Total spans for exercise having tokens: {0} {1}".format(args.label_prop_num, spans_with_tokens_less_than_5))
    print("Total spans for exercise: {0}".format(count))

    basename = os.path.basename(args.to_annotate).replace(".conll", "")
    LC_output_file = os.path.dirname(args.to_annotate) + "/" + basename + "_LC.conll"
    count_tokens = args.k
    with codecs.open(LC_output_file, "w", encoding='utf-8') as fout:
        idx = 0
        for sentence_key in sorted(score_sentence.keys(), reverse=False):
            if count_tokens<=0:
                break
            sent = sentence_key.split()[1:-1]
            sent = [data_loader.id_to_word[int(token)] for token in sent]
            gold_path = gold_dict[" ".join(sent)]
            token_count = 0
            for token in sent:
                count_tokens -= 1
                gold_tag_label = gold_path[token_count]


                token_count += 1

                fout.write(token+ "\t" + "UNK\t" + gold_tag_label + "\n")
            fout.write("\n")
            idx += 1


    word_weights = outputWordEmbedding(model, sentence_index)
    pkl.dump(word_weights, open(args.model_name + "_char_word_representations.pkl","wb"))
    print("Output word embeddings: ",args.model_name + "_char_word_representations.pkl")

def test_single_model(args):
    ner_data_loader = NER_DataLoader(args)
    test_pos_tags = []
    dev_pos_tags = []
    train_pos_tags_idx, dev_pos_tags_idx, test_pos_tags_idx = [], [], []
    if args.multilingual:
        pos_tagset = set()
        for lang in args.langs.split("/"):
            if args.use_pos:
                pos_folder = args.pos_folder + "/" + "UD_" + ner_data_loader.code_to_lang[lang] + "//"
                pos_train_file = pos_folder + "POS_train.conll"
                print("Reading from, ", pos_train_file)
                _ = ner_data_loader.get_pos_data_set(pos_train_file, pos_tagset)

        if args.use_pos:
            pos_folder = args.pos_folder + "/" + "UD_" + ner_data_loader.code_to_lang[args.lang] + "//"
            pos_dev_file = pos_folder + "POS_pred_dev.conll"
            lang_dev_pos_tags = ner_data_loader.get_pos_data_set(pos_dev_file, pos_tagset, isDev=True)
            dev_pos_tags += lang_dev_pos_tags

            if "test" in args.test_path:
                pos_test_file = pos_folder + "POS_test.conll"
                lang_test_pos_tags = ner_data_loader.get_pos_data_set(pos_test_file, pos_tagset, isDev=True)
            else:
                pos_test_file = pos_folder + "POS_train.conll"
                lang_test_pos_tags = ner_data_loader.get_pos_data_set(pos_test_file, pos_tagset, isDev=False)

            test_pos_tags += lang_test_pos_tags
            pos_tagset.add("_")
            ner_data_loader.pos_labe2idx = ner_data_loader.get_vocab_from_set(pos_tagset)
            ner_data_loader.pos_vocab_size = len(ner_data_loader.pos_labe2idx.keys())
            ner_data_loader.pos_idx2label = {v: k for k, v in ner_data_loader.pos_labe2idx.items()}
            ner_data_loader.pretrain_pos_emb = loadPretrainedEmbedding(args.pos_emb_file, args.pos_emb_dim,
                                                                       ner_data_loader.pos_labe2idx)
            for data_point in test_pos_tags:
                test_pos_tags_idx.append([ner_data_loader.pos_labe2idx[token] for token in data_point])

            for data_point in dev_pos_tags:
                dev_pos_tags_idx.append([ner_data_loader.pos_labe2idx[token] for token in data_point])


    else:
        _, _, _, _, _, _ = ner_data_loader.get_data_set(args.train_path, args.lang)


    print("Using Char Birnn Attn model!")
    model = BiRNN_ATTN_CRF_model(args, ner_data_loader)
    model.load()


    if args.visualize:
        #printTransitionMatrix(model, ner_data_loader, "POS")
        outputLanguageEmbedding(model, ner_data_loader)
    
    if args.test_conll:

        acc, precision, recall, f1  = finalEval(args, dev_pos_tags_idx, None, model, ner_data_loader, test_pos_tags_idx)

        pkl.dump(model.attention_weights, open("./attention_weights.pkl","wb"))
        #word_embeddings = outputWordEmbedding(model)
        #pkl.dump(word_embeddings, open(args.model_name + "_char_word_representations.pkl","wb"))
        #print("Output word embeddings: ",args.model_name + "_char_word_representations.pkl")

#Helper functions for Interpretabilty

def printTransitionMatrix(model, dataloader, tag):

    transition_matrix = dy.parameter(model.crf_decoders[tag].transition_matrix).npvalue() #(from, to)
    print(dataloader.id2tags[tag])
    plot_heatmap(transition_matrix[:-2,:-2], dataloader.id2tags[tag], tag)

    #np.save("pos_transition_matrix",transition_matrix)
    exit(-1)

def outputWordEmbedding(model, sentence_index):
    word_weights = []
    print("Number: ",len(model.word_embedding_weights))
    for sent_num in sentence_index:
        char_word_emb  = model.word_embedding_weights[sent_num]
        for char_emb in char_word_emb:
            word_weights.append(char_emb)
    return word_weights

def outputLanguageEmbedding(model, dataloader):
    with codecs.open("./learned_typolgoical_vectors.vec", "w", encoding='utf-8') as fvec:
        fvec.write("Learned typology weights \n")
        for lang, weight in model.typology_encoder.W.items():
            feature_vector = dataloader.pre_computed_features[lang.replace("<","").replace(">","")]
            W = dy.parameter(weight).npvalue()
            b= dy.parameter(model.typology_encoder.b[lang]).npvalue()
            print(W.shape, b.shape, feature_vector.shape)
            vector = np.dot(W, feature_vector) + b
            print(vector.shape)
            fvec.write(lang + "\t"  + " ".join(map(str,vector)) + "\n")
            lang_id = [dataloader.word_to_id[lang]]
            lang_emb = model.word_lookup.encode([lang_id])[0].npvalue()
            fvec.write(lang + "\t" + " ".join(map(str,lang_emb)))
            fvec.write("\n")

def plot_heatmap(weights, id_to_tag, tag):
    font = {'family': 'normal',
            'size': 14,
            'weight': 'bold'}

    matplotlib.rc('font', **font)

    # weights is a ParameterList
    tag_labels = [id_to_tag[id] for id in range(len(weights))]
    plt.figure(figsize=(20, 18), dpi=80)
    plt.xticks(range(0, len(tag_labels)), tag_labels, rotation=45)
    plt.yticks(range(0, len(tag_labels)), tag_labels)
    plt.tick_params(labelsize=40)
    plt.xlabel(tag, fontsize=50)
    plt.ylabel(tag, fontsize=50)
    plt.imshow(weights, cmap='Greys', interpolation='nearest')
    plt.savefig("./" + tag + "_" + tag + ".png", bbox_inches='tight')
    plt.close()


from args import init_config

args = init_config()
from models.model_builder import *
import uuid
from dataloaders.data_loader import *
uid = uuid.uuid4().get_hex()[:6]

if __name__ == "__main__":
    # args = init_config()
    if args.mode == "train":
        if args.load_from_path is not None:
            args.load_from_path = args.load_from_path
        else:
            args.load_from_path = args.save_to_path
        main(args)

    elif args.mode == "test_1":
        test_single_model(args)

    else:
        raise NotImplementedError
