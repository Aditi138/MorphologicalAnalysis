__author__ = 'chuntingzhou'
from encoders import *
from decoders import *
from collections import defaultdict
from copy import deepcopy
import codecs
from itertools import combinations
#np.set_printoptions(threshold='nan')
minx = -1
maxx = 1
from nltk.cluster import KMeansClusterer
import nltk
import scipy
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import cluster



class CRF_Model(object):
    def __init__(self, args, data_loader, lm_data_loader=None):
        self.save_to = args.save_to_path
        self.load_from = args.load_from_path
        tag_to_ids = data_loader.tag_to_ids
        self.constraints = None
        # print self.constraints
        self.lm_obj = args.lm_obj
        self.data_loader = data_loader


        #partial CRF
        self.use_partial = args.use_partial
        self.tag_to_ids = tag_to_ids
        self.id_to_tags = data_loader.id2tags
        self.B_UNK = data_loader.B_UNK


        #active learning for partial annotations
        self.use_CFB = args.use_CFB
        self.use_similar_label = args.use_similar_label
        self.sumType = args.sumType
        self.use_pos = args.use_pos
        self.use_typology = args.use_typology
        self.k = args.k
        self.cluster_nums = args.clusters
        self.activeLearning = args.activeLearning
        self.use_label_propagation = args.use_label_propagation

        self.entropy_spans = defaultdict(lambda: 0)
        self.most_uncertain_entropy_spans = {}
        self.full_sentences = defaultdict(list)
        self.avg_spans_in_sent_entropy = defaultdict(list)
        self.token_count = defaultdict(lambda:0)
        self.typeHeap = defaultdict(list)
        self.heapInfo = defaultdict(list)
        self.clusterDetails = args.clusterDetails
        self.token_index_entropy_map = {}
        self.to_index =  0


        if self.use_typology:
            self.pre_computed_features = data_loader.pre_computed_features

        self.fout = codecs.open(args.model_name  + "_featureWiseEntropy.txt","w", encoding='utf-8')

    def forward(self, sents, char_sents, feats, bc_feats, training=True):
        raise NotImplementedError

    def save(self):
        if self.save_to is not None:
            self.model.save(self.save_to)
        else:
            print('Save to path not provided!')

    def load(self, path=None):
        if path is None:
            path = self.load_from
        if self.load_from is not None or path is not None:
            print('Load model parameters from %s!' % path)
            self.model.populate(path)
        else:
            print('Load from path not provided!')

    def cal_loss(self, sents, char_sents, ner_tags, feats, bc_feats, known_tags, pos_tags=None, pre_computed_features=None, langs=None,lm_batch=None, training=True):
        birnn_outputs = self.forward(sents, char_sents, feats, bc_feats, pos_tags,pre_computed_features, langs, training=training)

        first = True
        for key, gold_tags in ner_tags.items():
            if first:
                neg_log_likelihoods = self.crf_decoders[key].decode_loss(birnn_outputs, gold_tags,self.use_partial, known_tags[key], self.tag_to_ids[key], self.B_UNK, self.B_UNK)
                first = False
            else:
                neg_log_likelihoods += self.crf_decoders[key].decode_loss(birnn_outputs, gold_tags, self.use_partial,
                                                                         known_tags[key], self.tag_to_ids[key], self.B_UNK,
                                                                         self.B_UNK)

        neg_log_likelihoods = neg_log_likelihoods / len(ner_tags)
        crf_loss = dy.sum_batches(neg_log_likelihoods) / len(sents)
        return crf_loss#, sum_s, sent_s

    def eval(self, sents, char_sents, feats, bc_feats,b_pos=None,  pre_computed_features=None,langs=None, training=False,type="dev"):
        birnn_outputs = self.forward(sents, char_sents, feats, bc_feats, b_pos, pre_computed_features, langs, training=training, type=type)

        best_scores, best_paths = {},{}
        featureEntropies = {}
        tokenEntropy = {}
        tokenInfo = {}
        sent = sents[0]

        #For each feature decoder
        for key, crf_decoder in self.crf_decoders.items():

            best_score, best_path, tag_scores = self.crf_decoders[key].decoding(birnn_outputs)
            best_scores[key] = best_score
            best_paths[key] = best_path
            best_path_copy = deepcopy(best_path)
            featureWiseEntropy = defaultdict(lambda:0)

            if type == "test" and self.activeLearning:
                alpha_value, alphas = crf_decoder.forward_alg(tag_scores)
                beta_value, betas = crf_decoder.backward_one_sequence(tag_scores)
                # print("Alpha:{0} Beta:{1}".format(alpha_value.value(), beta_value.value()))

                gammas = []
                for i in range(len(sent)):
                    gammas.append(alphas[i] + betas[i] - alpha_value)


                #Different active learning strategies
                if self.use_CFB: #Confidence Field Estimation (Culotta and McCallum 2004).
                    print("Using CFB")
                    crf_decoder.get_uncertain_subsequences_CFE(sent[1:-1], tag_scores[1:-1], best_path_copy, alpha_value, self.B_UNK, self.tag_to_ids[key],
                                                           self.entropy_spans, self.most_uncertain_entropy_spans,
                                                           self.full_sentences, self.avg_spans_in_sent_entropy)

                elif self.use_similar_label: #Entropy based method by accumulating similar tokens based on char-based representation.
                    crf_decoder.get_uncertain_subsequences_similar(sent[1:-1], gammas[1:-1],best_path_copy,
                                                       self.entropy_spans, self.most_uncertain_entropy_spans,featureWiseEntropy, self.SimilarLabels, tokenEntropy, tokenInfo)

                elif self.use_label_propagation: #ETAL + label_propgation
                    crf_decoder.get_uncertain_subsequences_labelProp(sent[1:-1], gammas[1:-1], best_path_copy,
                                                                     self.entropy_spans,
                                                                     self.most_uncertain_entropy_spans,
                                                                     featureWiseEntropy, tokenEntropy, tokenInfo)

                else: #Entropy based method by accumulating only on exact match.
                    crf_decoder.get_uncertain_subsequences_labelProp(sent[1:-1], gammas[1:-1], best_path_copy,
                                                                     self.entropy_spans,
                                                                     self.most_uncertain_entropy_spans,
                                                                     featureWiseEntropy, tokenEntropy, tokenInfo)

                featureEntropies[key] = featureWiseEntropy


        if  type== "test":
            for token_index, token in enumerate(sent[1:-1]):
                token_entropy = tokenEntropy[token_index]
                token_info = tokenInfo[token_index]
                self.typeHeap[str(token)].append(token_entropy)
                self.heapInfo[str(token)].append(token_info)
                self.token_index_entropy_map[self.to_index] = token_entropy
                self.to_index +=1


        #DEBUG - Printing entropy per feature
        # if type == "test":
        #     POS_entropies = featureEntropies["POS"]
        #     all_features = set(featureEntropies.keys()) - set(["POS"])
        #     for span in sent[1:-1]:
        #         self.fout.write(self.data_loader.id_to_word[int(span)] + "\t" + "POS" + ":" + str(POS_entropies[str(span)]) + "\t")
        #         for feature in all_features:
        #             self.fout.write( feature + ":" + str(featureEntropies[feature][str(span)]) + "\t")
        #         self.fout.write("\n")
        #     self.fout.write("\n")
        return  best_scores, best_paths

    def getscores(self, sents, char_sents, feats, bc_feats,b_pos=None,  pre_computed_features=None,langs=None, training=False,type="dev"):
        birnn_outputs = self.forward(sents, char_sents, feats, bc_feats, b_pos, pre_computed_features, langs, training=training,type=type)

    def getSimilarityMatrix(self, tgt_tags = None):
        # kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
        # assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

        if tgt_tags: #If gold tags are there
            gold_tags_token = {}
            for key, tags in tgt_tags.items():
                index = 0
                for sent_tag in tags:
                    for tag in sent_tag[1:-1]:
                        if index not in gold_tags_token:
                            gold_tags_token[index] = set()
                        gold_tags_token[index].add(self.id_to_tags[key][tag])
                        index +=1



        NUM_CLUSTERS = self.cluster_nums
        kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
        kmeans.fit(self.token_embeddings)
        centroids = kmeans.cluster_centers_
        self.SimilarLabels = kmeans.labels_

        # Get silhouette samples
        self.silhouette_vals = silhouette_samples(self.token_embeddings, self.SimilarLabels )
        self.token_indices = np.array([i for i in range(len(self.silhouette_vals))])

        clusterDetails  = []
        with codecs.open(self.clusterDetails, "w", encoding='utf-8') as fout:
            for i, cluster_num in enumerate(np.unique(self.SimilarLabels)):
                cluster_silhouette_vals = self.silhouette_vals[self.SimilarLabels == cluster_num]
                fout.write(str(cluster_num) + "\t" + str(len(cluster_silhouette_vals)) + "\n")
                clusterDetails.append(len(cluster_silhouette_vals))

            Max, Min, Mean, per = np.max(clusterDetails), np.min(clusterDetails), np.mean(clusterDetails), np.percentile(clusterDetails, 90)
            fout.write("Max: {0}, Min: {1}, Mean: {2}, 90Per: {3}".format(Max, Min, Mean, per))


        self.mostSimilar = defaultdict(list)
        self.labels_to_index = {}
        self.gold_labels_per_cluster = defaultdict(list)


        for index, label in enumerate(self.SimilarLabels):
            self.mostSimilar[label].append(self.tokens[index])
            if tgt_tags:
                self.gold_labels_per_cluster[label].append(";".join(list(gold_tags_token[index])))

    def eval_scores(self, sents, char_sents, feats, bc_feats, training=False):
        birnn_outputs = self.forward(sents, char_sents, feats, bc_feats, training=training)
        tag_scores, transit_score = self.crf_decoder.get_crf_scores(birnn_outputs)
        return tag_scores, transit_score


class BiRNN_ATTN_CRF_model(CRF_Model):
    def __init__(self, args, data_loader, lm_data_loader=None):
        self.model = dy.Model()
        self.args = args
        super(BiRNN_ATTN_CRF_model, self).__init__(args, data_loader)
        tag_vocab_sizes = data_loader.tag_vocab_sizes
        num_feats = len(tag_vocab_sizes)
        char_vocab_size = data_loader.char_vocab_size
        word_vocab_size = data_loader.word_vocab_size
        word_padding_token = data_loader.word_padding_token

        char_emb_dim = args.char_emb_dim
        word_emb_dim = args.word_emb_dim
        tag_emb_dim = args.tag_emb_dim

        if args.only_char:
            birnn_input_dim = args.char_hidden_dim * 2
        else:
            birnn_input_dim = args.char_hidden_dim * 2 + args.word_emb_dim
        hidden_dim = args.hidden_dim
        char_hidden_dim = args.char_hidden_dim
        self.char_hidden_dim = args.char_hidden_dim * 2
        src_ctx_dim = args.hidden_dim * 2

        output_dropout_rate = args.output_dropout_rate
        emb_dropout_rate = args.emb_dropout_rate


        self.char_birnn_encoder = BiRNN_Encoder(self.model,
                 char_emb_dim,
                 char_hidden_dim,
                 emb_dropout_rate=0.0,
                 output_dropout_rate=0.0,
                 vocab_size=char_vocab_size,
                 emb_size=char_emb_dim)

        self.proj1_W = self.model.add_parameters((char_hidden_dim, char_hidden_dim * 2))
        self.proj1_b = self.model.add_parameters(char_hidden_dim)
        self.proj1_b.zero()
        self.proj2_W = self.model.add_parameters((char_hidden_dim, char_hidden_dim * 2))
        self.proj2_b = self.model.add_parameters(char_hidden_dim)
        self.proj2_b.zero()

        if args.pretrain_emb_path is None:
            self.word_lookup = Lookup_Encoder(self.model, args, word_vocab_size, word_emb_dim, word_padding_token)
        else:
            print("In NER CRF: Using pretrained word embedding!")
            self.word_lookup = Lookup_Encoder(self.model, args, word_vocab_size, word_emb_dim, word_padding_token, data_loader.pretrain_word_emb)

        if args.use_char_attention:
            birnn_input_dim = birnn_input_dim + args.char_hidden_dim * 2

        self.char_birnn_modeling = BiRNN_Encoder(self.model,
                                                 args.char_hidden_dim * 4,
                                                 args.char_hidden_dim * 2,
                                                emb_dropout_rate=emb_dropout_rate,
                                                output_dropout_rate=output_dropout_rate)

        if args.use_pos:
            self.pos_emb_dim = args.pos_emb_dim
            self.pos_lookup = Lookup_Encoder(self.model, args, data_loader.pos_vocab_size, self.pos_emb_dim, word_padding_token, data_loader.pretrain_pos_emb)
            birnn_input_dim = birnn_input_dim + self.pos_emb_dim

        if args.use_token_langid:
            birnn_input_dim = birnn_input_dim + word_emb_dim

        if args.use_typology:

            # self.typology_encoder= Typology_Feature_Encoder(self.model, data_loader.feature_emb_dim,hidden_dim*2,  args)
            self.typology_encoder = Typology_Feature_Encoder(self.model, data_loader.feature_emb_dim, args, data_loader.id_to_word)
            src_ctx_dim = src_ctx_dim * args.typology_feature_dim

        self.birnn_encoder = BiRNN_Encoder(self.model,
                                           birnn_input_dim,
                                           hidden_dim,
                                           emb_dropout_rate=emb_dropout_rate,
                                           output_dropout_rate=output_dropout_rate)


        self.crf_decoders = {}
        for key, tag_size in tag_vocab_sizes.items():
            self.crf_decoders[key] = chain_CRF_decoder(args, self.model, src_ctx_dim, tag_emb_dim, tag_size, constraints=self.constraints)

        self.attention_weights = []
        self.word_embedding_weights = []
        self.token_embeddings = []
        self.tokens = []
        self.token_gold = []
        self.id_to_char = data_loader.id_to_char
        self.token_index_sent_index_map = {}
        self.t_index, self.s_index,self.sent_index_sent_map, self.token_index_relative_idx_map  = 0,0, {},{}

    def forward(self, sents, char_sents, feats, bc_feats, b_pos_tags, b_typo_features, b_langs, training=True, type="dev"):
        if self.args.use_char_attention:
            char_embs = self.char_birnn_encoder.encode_char(char_sents, training=training)
            proj1_W = dy.parameter(self.proj1_W)
            proj1_b = dy.parameter(self.proj1_b)
            proj2_W = dy.parameter(self.proj2_W)
            proj2_b = dy.parameter(self.proj2_b)
            attended_sents = []
            for i,batch in enumerate(char_embs):
                attended_sent = []
                for word_num,word in enumerate(batch):
                    E = np.ones((len(word), len(word)), dtype=float) - np.eye(len(word))
                    attn_keys = [dy.tanh(dy.affine_transform([proj1_b, proj1_W, w_attn])) for w_attn in word]
                    attn_values = dy.concatenate_cols([dy.tanh(dy.affine_transform([proj2_b, proj2_W, w_attn])) for w_attn in word])
                    attn_weights = [dy.softmax(dy.transpose(key) * attn_values, d=1) for key in attn_keys]
                    word_representation = dy.concatenate_cols(word)

                    if not training:
                        self.attention_weights.append([a.npvalue() for a in attn_weights])
                        #orig_word = char_sents[i][word_num]
                        #self.attn_fout.write("".join(self.id_to_char[id] for id in orig_word) + "\t" + " ".join(map(str,attn_weights_value)) + "\n")


                    maskedAttention =[]
                    for j in range(len(word)):
                        masking = dy.cmult(dy.inputTensor(np.concatenate([E[j] for _ in range(self.char_hidden_dim)]).reshape((self.char_hidden_dim, len(word)))),
                             word_representation)
                        maskedAttention.append(masking * dy.transpose(attn_weights[j]))

                    attended_sent.append([dy.concatenate([h, ha]) for h, ha in zip(word, maskedAttention)])
                    # attended_sent.append(dy.mean_dim(
                    #     dy.concatenate_cols([dy.concatenate([h, ha]) for h, ha in zip(word, maskedAttention)]),
                    #     d=[1], b=False))
                attended_sents.append(attended_sent)

                char_embs = self.char_birnn_modeling.encode(attended_sents, training=training, char=True, model_char=True)
                if not training and type == "test":
                    self.word_embedding_weights.append([c.npvalue() for c in char_embs[1:-1]])
                    for c in char_embs[1:-1]:
                        self.token_embeddings.append(c.npvalue())
                    for rel_idx, token in enumerate(sents[0][1:-1]):
                        self.tokens.append(token)
                        self.token_index_sent_index_map[self.t_index] = self.s_index
                        self.token_index_relative_idx_map[self.t_index] = rel_idx
                        self.t_index +=1

                    self.sent_index_sent_map[self.s_index] = sents[0][1:-1]
                    self.s_index += 1

            #char_embs, _ = transpose_and_batch_embs(attended_sents,
            #                                                  self.char_hidden_dim * 2)  # [(hidden_dim*2, batch_size)]
        else:
            char_embs = self.char_birnn_encoder.encode(char_sents, training=training, char=True)
        
        word_embs = self.word_lookup.encode(sents)

        if self.args.use_token_langid:
            lang_embs = self.word_lookup.encode(b_langs)

        if self.args.use_pos:
            b_pos_embs = self.pos_lookup.encode(b_pos_tags)


        if self.args.only_char: #not appending word embeddings

            if self.args.use_pos and self.args.use_token_langid:
                concat_inputs = [dy.concatenate([c, f, l]) for c, f, l in
                                 zip(char_embs, b_pos_embs, lang_embs)]

            elif self.args.use_pos:
                concat_inputs = [dy.concatenate([c,f]) for c, f in
                                 zip(char_embs, b_pos_embs)]

            elif self.args.use_token_langid:
                concat_inputs = [dy.concatenate([c, l]) for c, l in
                                 zip(char_embs, lang_embs)]
            else:
                concat_inputs = [dy.concatenate([c]) for c in char_embs]

        else:

            if self.args.use_pos and self.args.use_token_langid:
                concat_inputs = [dy.concatenate([c, w, f,l]) for c, w, f,l in
                                 zip(char_embs, word_embs, b_pos_embs,lang_embs)]

            elif self.args.use_pos:
                concat_inputs = [dy.concatenate([c, w, f]) for c, w, f in
                                 zip(char_embs, word_embs, b_pos_embs)]

            elif self.args.use_token_langid:
                concat_inputs = [dy.concatenate([c, w, l]) for c, w, l in
                                 zip(char_embs, word_embs, lang_embs)]

            else:
                concat_inputs = [dy.concatenate([c, w]) for c, w in zip(char_embs, word_embs)]

        birnn_outputs = self.birnn_encoder.encode(concat_inputs, training=training)

        return birnn_outputs

