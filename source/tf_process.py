import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import source.utils as utils

def measure_auroc(positive, negative):

    y_label, y_score = None, None

    y_label = list(np.ones(len(positive)))
    y_label.extend(list(np.zeros(len(negative))))

    y_score = positive
    y_score.extend(list(negative))

    fpr, tpr, thresholds = roc_curve(y_label, y_score, pos_label=1)
    auroc = auc(fpr, tpr)

    return auroc

def perform_from_confmat(confusion_matrix, num_class, verbose=False):

    dict_perform = {'accuracy':0, 'precision':0, 'recall':0, 'f1score':0}

    for idx_c in range(num_class):
        precision = np.nan_to_num(confusion_matrix[idx_c, idx_c] / np.sum(confusion_matrix[:, idx_c]))
        recall = np.nan_to_num(confusion_matrix[idx_c, idx_c] / np.sum(confusion_matrix[idx_c, :]))
        f1socre = np.nan_to_num(2 * (precision * recall / (precision + recall)))

        dict_perform['accuracy'] += confusion_matrix[idx_c, idx_c]
        dict_perform['precision'] += precision
        dict_perform['recall'] += recall
        dict_perform['f1score'] += f1socre

        if(verbose):
            print("Class-%d | Precision: %.5f, Recall: %.5f, F1-Score: %.5f" \
                %(idx_c, precision, recall, f1socre))

    for key in list(dict_perform.keys()):
        if('accuracy' == key): dict_perform[key] = dict_perform[key] / np.sum(confusion_matrix)
        else: dict_perform[key] = dict_perform[key] / num_class

    return dict_perform

def training(agent, dataset, batch_size, epochs):

    print("\n** Training of the AE to %d epoch | Batch size: %d" %(epochs, batch_size))
    iteration = 0

    for epoch in range(epochs):

        while(True):
            minibatch, terminate = dataset.next_batch(batch_size=batch_size, ttv=0)
            if(len(minibatch.keys()) == 0): break
            step_dict = agent.step(minibatch=minibatch, iteration=iteration, training=True)
            iteration += 1
            if(terminate): break
        dataset.reset_index()

        print("Epoch [%d / %d] | Loss: %f" %(epoch, epochs, step_dict['losses']['entropy']))
        agent.save_params(model='model_0_finepocch')

    for idx_k, name_key in enumerate(list(best_val.keys())):
        print(name_key, best_val[name_key])

def test(agent, dataset, batch_size):

    savedir = 'results_te'
    utils.make_dir(path=savedir, refresh=True)

    list_model = utils.sorted_list(os.path.join('Checkpoint', 'model*'))
    for idx_model, path_model in enumerate(list_model):
        list_model[idx_model] = path_model.split('/')[-1]

    for idx_model, path_model in enumerate(list_model):

        print("\n** Test with %s" %(path_model))
        agent.load_params(model=path_model)
        utils.make_dir(path=os.path.join(savedir, path_model), refresh=False)

        fcsv = open(os.path.join(savedir, path_model, 'scores.csv'), 'w')
        fcsv.write("label,score0\n")
        confusion_matrix = np.zeros((dataset.num_class, dataset.num_class), np.int32)
        while(True):
            minibatch, terminate = dataset.next_batch(batch_size=batch_size, ttv=1)
            if(len(minibatch.keys()) == 0): break
            step_dict = agent.step(minibatch=minibatch, training=False)

            for idx_y, _ in enumerate(minibatch['label_1hot']):
                confusion_matrix[np.argmax(minibatch['label_1hot'][idx_y]), np.argmax(step_dict['y_hat'][idx_y])] += 1
                fcsv.write("%d,%f\n" %(minibatch['label'][idx_y][-1], step_dict['y_hat'][idx_y][0]))

            if(terminate): break
        dataset.reset_index()

        dict_perform = perform_from_confmat(confusion_matrix=confusion_matrix, num_class=dataset.num_class, verbose=True)
        fcsv.close()
        np.save(os.path.join(savedir, path_model, 'conf_mat.npy'), confusion_matrix)

        df_score = pd.read_csv(os.path.join(savedir, path_model, 'scores.csv'))
        df_pos = df_score[df_score['label'] == 1]
        df_neg = df_score[df_score['label'] == -1]
        auroc = measure_auroc(list(df_pos['score0']), list(df_neg['score0']))

        print(path_model, auroc)
