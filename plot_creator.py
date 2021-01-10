import matplotlib.pyplot as plt
import numpy as np


def auto_encoder_plots(root_dir, datasets, epsilon=1e-6, save=False):
    eval_str = 'eval_perf_'
    perf_str = 'train_perf_'
    eval_files = []
    perf_files = []
    for dataset in datasets:
        eval_files.append(np.load(root_dir + eval_str + dataset + '_200_autoencoder_net.npy'))
        perf_files.append(np.load(root_dir + perf_str + dataset + '_200_autoencoder_net.npy'))

    plt.figure(figsize=(20, 20))
    if not save:
        plt.suptitle('AutoEncoder Training Results')
    for index, dataset in enumerate(datasets):
        plt.subplot(3, 3, index + 1)
        # Accuracy
        x = np.arange(0, 200)
        param_1 = perf_files[index][7]
        param_2 = perf_files[index][19]
        plt.plot(x, param_1, x, param_2)
        plt.title(dataset)
        plt.legend(['Training', 'Validation'])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
    if save:
        plt.savefig('AutoEncoder_Accuracy.png', bbox_inches='tight')
    else:
        plt.show()

    plt.figure(figsize=(20, 20))
    if not save:
        plt.suptitle('AutoEncoder Training Results')
    for index, dataset in enumerate(datasets):
        plt.subplot(3, 3, index+1)
        # AuC
        x = np.arange(0, 200)
        param_1 = perf_files[index][10]
        param_2 = perf_files[index][22]
        plt.plot(x, param_1, x, param_2)
        plt.title(dataset)
        plt.legend(['Training', 'Validation'])
        plt.xlabel('Epochs')
        plt.ylabel('AuC')
    if save:
        plt.savefig('AutoEncoder_AUC.png', bbox_inches='tight')
    else:
        plt.show()

    plt.figure(figsize=(20, 20))
    if not save:
        plt.suptitle('AutoEncoder Training Results')
    for index, dataset in enumerate(datasets):
        plt.subplot(3, 3, index + 1)
        # F-1 Score
        x = np.arange(0, 200)
        precision_train, precision_val = perf_files[index][8], perf_files[index][20]
        recall_train, recall_val = perf_files[index][9], perf_files[index][21]
        param_1 = (2 * precision_train * recall_train) / (precision_train + recall_train + epsilon)
        param_2 = (2 * precision_val * recall_val) / (precision_val + recall_val + epsilon)
        plt.plot(x, param_1, x, param_2)
        plt.title(dataset)
        plt.legend(['Training', 'Validation'])
        plt.xlabel('Epochs')
        plt.ylabel('F-1 Score')
    if save:
        plt.savefig('AutoEncoder_F1Score.png', bbox_inches='tight')
    else:
        plt.show()

    """# List of outputs:
    0 loss ? 
    1 label_loss
    2 autogen_loss
    3 true_positives
    4 false_positives
    5 true_negatives
    6 false_negatives
    7 categorical accuracy
    8 precision
    9 recall
    10 auc
    11 cosine_similarity"""


def inception_net_plots(root_dir, datasets, epsilon=1e-6, save=False):
    eval_str = 'eval_perf_'
    perf_str = 'train_perf_'
    perf_files = []
    for dataset in datasets:
        perf_files.append(np.load(root_dir + perf_str + dataset + '_200_inception_net.npy'))

    # Accuracy
    plt.figure(figsize=(20, 20))
    if not save:
        plt.suptitle('InceptionNet Training Results')
    for index, dataset in enumerate(datasets):
        plt.subplot(3, 3, index + 1)
        x = np.arange(0, 200)
        param_1 = perf_files[index][5]
        param_2 = perf_files[index][14]
        plt.plot(x, param_1, x, param_2)
        plt.title(dataset)
        plt.legend(['Training', 'Validation'])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
    if save:
        plt.savefig('Inception_Accuracy.png', bbox_inches='tight')
    else:
        plt.show()

    # AuC
    plt.figure(figsize=(20, 20))
    if not save:
        plt.suptitle('InceptionNet Training Results')
    for index, dataset in enumerate(datasets):
        plt.subplot(3, 3, index + 1)
        x = np.arange(0, 200)
        param_1 = perf_files[index][8]
        param_2 = perf_files[index][17]
        plt.plot(x, param_1, x, param_2)
        plt.title(dataset)
        plt.legend(['Training', 'Validation'])
        plt.xlabel('Epochs')
        plt.ylabel('AuC')
    if save:
        plt.savefig('Inception_AUC.png', bbox_inches='tight')
    else:
        plt.show()

    # F-1 Score
    plt.figure(figsize=(20, 20))
    if not save:
        plt.suptitle('InceptionNet Training Results')
    for index, dataset in enumerate(datasets):
        plt.subplot(3, 3, index + 1)
        x = np.arange(0, 200)
        precision_train, precision_val = perf_files[index][6], perf_files[index][15]
        recall_train, recall_val = perf_files[index][7], perf_files[index][16]
        param_1 = (2 * precision_train * recall_train) / (precision_train + recall_train + epsilon)
        param_2 = (2 * precision_val * recall_val) / (precision_val + recall_val + epsilon)
        plt.plot(x, param_1, x, param_2)
        plt.title(dataset)
        plt.legend(['Training', 'Validation'])
        plt.xlabel('Epochs')
        plt.ylabel('F-1 Score')
    if save:
        plt.savefig('Inception_F1Score.png', bbox_inches='tight')
    else:
        plt.show()

    """# List of outputs:
        0 loss ? 
        1 true_positives
        2 false_positives
        3 true_negatives
        4 false_negatives
        5 categorical accuracy
        6 precision
        7 recall
        8 auc"""

    print(perf_files[0].shape)


def residual_net_plots(root_dir, datasets, epsilon=1e-6, save=False):
    eval_str = 'eval_perf_'
    perf_str = 'train_perf_'
    perf_files = []
    for dataset in datasets:
        perf_files.append(np.load(root_dir + perf_str + dataset + '_200_residual_net.npy'))

    # Accuracy
    plt.figure(figsize=(20, 20))
    if not save:
        plt.suptitle('ResidualNet Training Results')
    for index, dataset in enumerate(datasets):
        plt.subplot(3, 3, index + 1)
        x = np.arange(0, 200)
        param_1 = perf_files[index][5]
        param_2 = perf_files[index][14]
        plt.plot(x, param_1, x, param_2)
        plt.title(dataset)
        plt.legend(['Training', 'Validation'])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
    if save:
        plt.savefig('Residual_Accuracy.png', bbox_inches='tight')
    else:
        plt.show()

    # AuC
    plt.figure(figsize=(20, 20))
    if not save:
        plt.suptitle('ResidualNet Training Results')
    for index, dataset in enumerate(datasets):
        plt.subplot(3, 3, index + 1)
        x = np.arange(0, 200)
        param_1 = perf_files[index][8]
        param_2 = perf_files[index][17]
        plt.plot(x, param_1, x, param_2)
        plt.title(dataset)
        plt.legend(['Training', 'Validation'])
        plt.xlabel('Epochs')
        plt.ylabel('AuC')
    if save:
        plt.savefig('Residual_AUC.png', bbox_inches='tight')
    else:
        plt.show()

    # F-1 Score
    plt.figure(figsize=(20, 20))
    if not save:
        plt.suptitle('ResidualNet Training Results')
    for index, dataset in enumerate(datasets):
        plt.subplot(3, 3, index + 1)
        x = np.arange(0, 200)
        precision_train, precision_val = perf_files[index][6], perf_files[index][15]
        recall_train, recall_val = perf_files[index][7], perf_files[index][16]
        param_1 = (2 * precision_train * recall_train) / (precision_train + recall_train + epsilon)
        param_2 = (2 * precision_val * recall_val) / (precision_val + recall_val + epsilon)
        plt.plot(x, param_1, x, param_2)
        plt.title(dataset)
        plt.legend(['Training', 'Validation'])
        plt.xlabel('Epochs')
        plt.ylabel('F-1 Score')
    if save:
        plt.savefig('Residual_F-1Score.png', bbox_inches='tight')
    else:
        plt.show()
    """# List of outputs:
        0 loss ? 
        1 true_positives
        2 false_positives
        3 true_negatives
        4 false_negatives
        5 categorical accuracy
        6 precision
        7 recall
        8 auc"""


def dense_net_plots(root_dir, datasets, epsilon=1e-6, save=False):
    perf_str = 'train_perf_'
    perf_files = []
    for dataset in datasets:
        perf_files.append(np.load(root_dir + perf_str + dataset + '_200_dense_net.npy'))

    # Accuracy
    plt.figure(figsize=(20, 20))
    if not save:
        plt.suptitle('DenseNet Training Results')
    for index, dataset in enumerate(datasets):
        plt.subplot(3, 3, index + 1)
        x = np.arange(0, 200)
        param_1 = perf_files[index][5]
        param_2 = perf_files[index][14]
        plt.plot(x, param_1, x, param_2)
        plt.title(dataset)
        plt.legend(['Training', 'Validation'])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
    if save:
        plt.savefig('Dense_Accuracy.png', bbox_inches='tight')
    else:
        plt.show()

    # AuC
    plt.figure(figsize=(20, 20))
    if not save:
        plt.suptitle('DenseNet Training Results')
    for index, dataset in enumerate(datasets):
        plt.subplot(3, 3, index + 1)
        x = np.arange(0, 200)
        param_1 = perf_files[index][8]
        param_2 = perf_files[index][17]
        plt.plot(x, param_1, x, param_2)
        plt.title(dataset)
        plt.legend(['Training', 'Validation'])
        plt.xlabel('Epochs')
        plt.ylabel('AuC')
    if save:
        plt.savefig('Dense_AUC.png', bbox_inches='tight')
    else:
        plt.show()

    # F-1 Score
    plt.figure(figsize=(20, 20))
    if not save:
        plt.suptitle('DenseNet Training Results')
    for index, dataset in enumerate(datasets):
        plt.subplot(3, 3, index + 1)
        x = np.arange(0, 200)
        precision_train, precision_val = perf_files[index][6], perf_files[index][15]
        recall_train, recall_val = perf_files[index][7], perf_files[index][16]
        param_1 = (2 * precision_train * recall_train) / (precision_train + recall_train + epsilon)
        param_2 = (2 * precision_val * recall_val) / (precision_val + recall_val + epsilon)
        plt.plot(x, param_1, x, param_2)
        plt.title(dataset)
        plt.legend(['Training', 'Validation'])
        plt.xlabel('Epochs')
        plt.ylabel('F-1 Score')
    if save:
        plt.savefig('Dense_F1Score.png', bbox_inches='tight')
    else:
        plt.show()

    """# List of outputs:
        0 loss ? 
        1 true_positives
        2 false_positives
        3 true_negatives
        4 false_negatives
        5 categorical accuracy
        6 precision
        7 recall
        8 auc"""


def eval_plots(roots, datasets, epsilon=1e-6):

    accuracy = np.zeros((len(roots), len(datasets)))
    auc = np.zeros((len(roots), len(datasets)))
    f1 = np.zeros((len(roots), len(datasets)))

    for r_index, root in enumerate(roots):
        for index, dataset in enumerate(datasets):
            eval_file = np.load(root + '/eval_perf_' + dataset + '_200_' + root + '.npy')
            if root == 'autoencoder_net':
                accuracy[r_index, index] = eval_file[7]
                auc[r_index, index] = eval_file[10]
                precision_val, recall_val = eval_file[8], eval_file[9]
                f1[r_index, index] = (2 * precision_val * recall_val) / (precision_val + recall_val + epsilon)
            else:
                accuracy[r_index, index] = eval_file[5]
                auc[r_index, index] = eval_file[8]
                precision_val, recall_val = eval_file[6], eval_file[7]
                f1[r_index, index] = (2 * precision_val * recall_val) / (precision_val + recall_val + epsilon)

    x = np.arange(0, len(datasets))
    plt.figure(figsize=(20, 20))
    plt.subplot(3, 1, 1)
    plt.xticks(x, datasets)
    plt.ylabel('Accuracy')
    plt.title('Evaluation Results - Accuracy')
    plt.plot(x, accuracy[0, :], x, accuracy[1, :], x, accuracy[2, :], x, accuracy[3, :])
    plt.legend(['AutoEncoder', 'InceptionNet', 'ResidualNet', 'DenseNet'])
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.xticks(x, datasets)
    plt.title('Evaluation Results - Area Under Curve')
    plt.plot(x, auc[0, :], x, auc[1, :], x, auc[2, :], x, auc[3, :])
    plt.ylabel('AuC')
    plt.legend(['AutoEncoder', 'InceptionNet', 'ResidualNet', 'DenseNet'])
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.xticks(x, datasets)
    plt.title('Evaluation Results - F1 Score')
    plt.plot(x, f1[0, :], x, f1[1, :], x, f1[2, :], x, f1[3, :])
    plt.ylabel('F1 Score')
    plt.legend(['AutoEncoder', 'InceptionNet', 'ResidualNet', 'DenseNet'])
    plt.grid()
    plt.savefig('Evaluation Results', bbox_inches='tight')


datasets_ = ['AbnormalHeartbeat', 'Phoneme', 'ArrowHead', 'FaceAll', 'Beef', 'EthanolLevel', 'Strawberry',
             'Earthquakes', 'InlineSkate']
roots_ = ['autoencoder_net', 'inception_net', 'residual_net', 'dense_net']

# auto_encoder_plots('autoencoder_net/', datasets_, save=True)
# inception_net_plots('inception_net/', datasets_, save=True)
# residual_net_plots('residual_net/', datasets_, save=True)
# dense_net_plots('dense_net/', datasets_, save=True)

# eval_plots(roots_, datasets_)

data = np.load('inception_net/train_perf_InlineSkate_200_inception_net.npy')
epsilon = 1e-6
plt.figure(figsize=(20, 20))
plt.subplot(3, 1, 1)
x = np.arange(0, 200)
plt.plot(x, data[5], x, data[14])
plt.title('InlineSkate - InceptionNet - Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'])
plt.grid()
plt.subplot(3, 1, 2)
plt.grid()
x = np.arange(0, 200)
plt.plot(x, data[8], x, data[17])
plt.title('InlineSkate - InceptionNet - Area Under Curve')
plt.xlabel('Epochs')
plt.ylabel('AuC')
plt.legend(['Training', 'Validation'])
plt.subplot(3, 1, 3)
plt.grid()
x = np.arange(0, 200)
plt.xlabel('Epochs')
plt.ylabel('F1-Score')
plt.title('InlineSkate - InceptionNet - F1 Score')
precision_train, precision_val = data[6], data[15]
recall_train, recall_val = data[7], data[16]
param_1 = (2 * precision_train * recall_train) / (precision_train + recall_train + epsilon)
param_2 = (2 * precision_val * recall_val) / (precision_val + recall_val + epsilon)
plt.plot(x, param_1, x, param_2)
plt.legend(['Training', 'Validation'])
plt.savefig('Inception_InlineSkate', bbox_inches='tight')

