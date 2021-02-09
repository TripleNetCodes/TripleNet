import numpy as np
import params
from dataset import Reader
# import utils
from create_batch import get_pair_batch_train, get_pair_batch_test, toarray, get_pair_batch_train_common, toarray_float
import torch
from model import BiLSTM_Attention
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import os
import logging
import math
from matplotlib import pyplot as plt


def Triplenet(ratio, kkkk, data):
    # Dataset parameters
    params.batch_size = 2048
    params.lam2 = 1.0
    params.lam1 = 0
    data_path = data
    params.num_neighbor = 5
    params.anomaly_ratio = ratio
    model_name = "TripleNet_local_2048"
    if data_path == params.data_dir_DBPEDIA:
        data_name = "DBpedia"
        params.batch_size = 3600
    else:
        data_name = "NELL"

    dataset = Reader(data_path, isInjectTopK=False)
    all_triples = dataset.train_data
    labels = dataset.labels
    train_idx = list(range(len(all_triples) // 2))
    num_iterations = math.ceil(dataset.num_triples_with_anomalies / params.batch_size)
    total_num_anomalies = dataset.num_anomalies
    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler(os.path.join(params.log_folder,
                                                    model_name + "_" + data_name + "_" + str(ratio) + "_" + "_log.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logging.info('There are %d Triples with %d anomalies in the graph.' % (len(dataset.labels), total_num_anomalies))

    params.total_ent = dataset.num_entity
    params.total_rel = dataset.num_relation

    model_saved_path = model_name + "_" + data_name + "_" + str(ratio) + ".ckpt"
    model_saved_path = os.path.join(params.out_folder, model_saved_path)
    # model.load_state_dict(torch.load(model_saved_path))
    # Model BiLSTM_Attention
    model = BiLSTM_Attention(params.input_size_lstm, params.hidden_size_lstm, params.num_layers_lstm, params.dropout,
                             params.alpha).to(params.device)
    criterion = nn.MarginRankingLoss(params.gama)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    #
    for k in range(kkkk):
        for it in range(num_iterations):
            batch_h, batch_r, batch_t, batch_size = get_pair_batch_train_common(dataset, it, train_idx,
                                                                                params.batch_size,
                                                                                params.num_neighbor)
            batch_h = torch.LongTensor(batch_h).to(params.device)
            batch_t = torch.LongTensor(batch_t).to(params.device)
            batch_r = torch.LongTensor(batch_r).to(params.device)
            # input_triple, batch_size = get_pair_batch_train_common(dataset, it, train_idx,
            #                                                                     params.batch_size,
            #                                                                     params.num_neighbor)
            # input_triple = Variable(torch.LongTensor(input_triple).cuda())
            # batch_size = input_triples.size(0)
            out, out_att = model(batch_h, batch_r, batch_t)
            out = out.reshape(batch_size, -1, 2 * 3 * params.BiLSTM_hidden_size)
            out_att = out_att.reshape(batch_size, -1, 2 * 3 * params.BiLSTM_hidden_size)

            pos_h = out[:, 0, :]
            pos_z0 = out_att[:, 0, :]
            # pos_z1 = out_att[:, 1, :]
            neg_h = out[:, 1, :]
            neg_z0 = out_att[:, 1, :]
            # neg_z1 = out_att[:, 3, :]

            # loss function
            # positive
            pos_loss = params.lam1 * torch.norm(pos_h - pos_z0, p=2, dim=1) + \
                       params.lam2 * torch.norm(pos_h[:, 0:2 * params.BiLSTM_hidden_size] +
                                  pos_h[:, 2 * params.BiLSTM_hidden_size:2 * 2 * params.BiLSTM_hidden_size] -
                                  pos_h[:, 2 * 2 * params.BiLSTM_hidden_size:2 * 3 * params.BiLSTM_hidden_size], p=2,
                                  dim=1)
            # negative
            neg_loss = params.lam1 * torch.norm(neg_h - neg_z0, p=2, dim=1) + \
                       params.lam2 * torch.norm(neg_h[:, 0:2 * params.BiLSTM_hidden_size] +
                                  neg_h[:, 2 * params.BiLSTM_hidden_size:2 * 2 * params.BiLSTM_hidden_size] -
                                  neg_h[:, 2 * 2 * params.BiLSTM_hidden_size:2 * 3 * params.BiLSTM_hidden_size], p=2,
                                  dim=1)

            y = -torch.ones(batch_size).to(params.device)
            loss = criterion(pos_loss, neg_loss, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pos_loss_value = torch.sum(pos_loss) / (batch_size * 2.0)
            neg_loss_value = torch.sum(neg_loss) / (batch_size * 2.0)
            logging.info('There are %d Triples in this batch.' % batch_size)
            logging.info('Epoch: %d-%d, pos_loss: %f, neg_loss: %f, Loss: %f' % (
                k, it + 1, pos_loss_value.item(), neg_loss_value.item(), loss.item()))

            torch.save(model.state_dict(), model_saved_path)
    # # #
    # dataset = Reader(data_path, "test")
    model1 = BiLSTM_Attention(params.input_size_lstm, params.hidden_size_lstm, params.num_layers_lstm, params.dropout,
                              params.alpha).to(params.device)
    model1.load_state_dict(torch.load(model_saved_path))
    model1.eval()
    with torch.no_grad():
        all_loss = []
        all_label = []
        start_id = 0

        for i in range(num_iterations):
            batch_h, batch_r, batch_t, labels, start_id, batch_size = get_pair_batch_test(dataset, params.batch_size,
                                                                                          params.num_neighbor, start_id)
            # labels = labels.unsqueeze(1)
            # batch_size = input_triples.size(0)
            batch_h = torch.LongTensor(batch_h).to(params.device)
            batch_t = torch.LongTensor(batch_t).to(params.device)
            batch_r = torch.LongTensor(batch_r).to(params.device)
            labels = labels.to(params.device)
            out, out_att = model1(batch_h, batch_r, batch_t)

            # out_att = out_att.reshape(batch_size, 2, 2 * 3 * params.BiLSTM_hidden_size)
            out_att_view0 = out_att
            # out_att_view1 = out_att[:, 1, :]
            # [B, 600] [B, 600]

            loss = params.lam1 * torch.norm(out_att_view0 - out, p=2, dim=1) + \
                   params.lam2 * torch.norm(out[:, 0:2 * params.BiLSTM_hidden_size] +
                              out[:, 2 * params.BiLSTM_hidden_size:2 * 2 * params.BiLSTM_hidden_size] -
                              out[:, 2 * 2 * params.BiLSTM_hidden_size:2 * 3 * params.BiLSTM_hidden_size], p=2, dim=1)

            all_loss += loss
            all_label += labels

            logging.info('[Train] Evaluation on %d batch of Original graph' % i)

        total_num = len(all_label)

        # print("Total number of test tirples: ", total_num)
        AUC11 = roc_auc_score(toarray(all_label), toarray_float(all_loss))
        logging.info('[Train] AUC of %d triples: %f' % (total_num, AUC11))

        ratios = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
        for i in range(len(ratios)):
            num_k = int(ratios[i] * dataset.num_original_triples)
            top_loss, top_indices = torch.topk(toarray_float(all_loss), num_k, largest=True, sorted=True)
            top_labels = toarray([all_label[top_indices[iii]] for iii in range(len(top_indices))])
            top_sum = top_labels.sum()
            recall = top_sum * 1.0 / total_num_anomalies
            precision = top_sum * 1.0 / num_k

            logging.info('[Train][%s][%s] Precision %f -- %f : %f' % (data_name, model_name, ratio, ratios[i], precision))
            logging.info('[Train][%s][%s] Recall  %f-- %f : %f' % (data_name, model_name, ratio, ratios[i], recall))
            logging.info('[Train][%s][%s] anomalies in total: %d -- discovered:%d -- K : %d' % (
            data_name, model_name, total_num_anomalies, top_sum, num_k))

            if top_sum.item() < num_k and top_sum.item() != 0:
                # print(top_sum.item(), num_k)
                AUC_K = roc_auc_score(toarray(top_labels), toarray_float(top_loss))
                logging.info('[Train][%s][%s] xxxxxxxxxxxxxxxxxxxxxxx AUC %f -- %f : %f' % (
                    data_name, model_name, ratio, ratios[i], AUC_K))

        max_top_k = total_num_anomalies * 2
        # min_top_k = total_num_anomalies // 10

        top_loss, top_indices = torch.topk(toarray_float(all_loss), max_top_k, largest=True, sorted=True)
        top_labels = toarray([all_label[top_indices[iii]] for iii in range(len(top_indices))])

        anomaly_discovered = []
        for i in range(max_top_k):
            if i == 0:
                anomaly_discovered.append(top_labels[i])
            else:
                anomaly_discovered.append(anomaly_discovered[i-1] + top_labels[i])

        results_interval_10 = np.array([anomaly_discovered[i * 10] for i in range(max_top_k // 10)])

        logging.info('[Train] final results: %s' % str(results_interval_10))

        top_k = np.arange(1, max_top_k + 1)

        assert len(top_k) == len(anomaly_discovered), 'The size of result list is wrong'

        precision_k = np.array(anomaly_discovered) / top_k
        recall_k = np.array(anomaly_discovered) * 1.0 / total_num_anomalies

        precision_interval_10 = [precision_k[i * 10] for i in range(max_top_k // 10)]
        # print(precision_interval_10)
        logging.info('[Train] final Precision: %s' % str(precision_interval_10))
        recall_interval_10 = [recall_k[i * 10] for i in range(max_top_k // 10)]
        # print(recall_interval_10)
        logging.info('[Train] final Recall: %s' % str(recall_interval_10))


# anomaly_injected_ratios = [0.01, 0.05, 0.10, 0.001, 0.025, 0.075]
num_epochs = [1, 2, 3, 5]
anomaly_injected_ratios = [0.05]
dataset = [params.data_dir_DBPEDIA]

for num in num_epochs:
    for ratio in anomaly_injected_ratios:
        for data in dataset:
            Triplenet(ratio, num, data)

