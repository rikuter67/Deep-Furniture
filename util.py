import argparse
import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np
import os
import pickle
from sklearn.manifold import TSNE
import pdb

def parser_run(): # parser for run.py
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, choices=['maxPooling', 'poolingMA', 'CSS', 'setRepVec_biPMA', 'setRepVec_pivot'], default='setRepVec_pivot', help='mode of computing set-matching score')
    parser.add_argument('-baseChn', type=int, default=64, help='number of base channel, default=32') # 次元数の半分？
    parser.add_argument('-model', type=str, default='VLAD', choices=['VLAD', 'SMN', 'random', 'CNN'], help='model type, default=VLAD')
    parser.add_argument('-num_layers', type=int, default=3, help='number of layers (attentions) in encoder and decoder, default=3')
    parser.add_argument('-num_heads', type=int, default=5, help='number of heads in attention, default=5')
    parser.add_argument('-is_set_norm', type=int, default=1, help='switch of set-normalization (1:on, 0:off), default=1')
    parser.add_argument('-is_cross_norm', type=int, default=1, help='switch of cross-normalization (1:on, 0:off), default=1')
    parser.add_argument('-trial', type=int, default=1, help='index of trial, default=1')
    parser.add_argument('-calc_set_sim', type=str, choices=['CS', 'BERTscore'], default='CS', help='how to evaluate set similarity')
    parser.add_argument('-use_Cvec', type=bool, default=True, help='Whether use Cvec')
    parser.add_argument('-is_Cvec_linear', type=bool, default=False, help='Whether learn FC_projection for Cluster seed vec') # 4096次元の候補ベクトルの次元削減FC層を学習するか否か？
    parser.add_argument('-year', type=int, default=2017, help='year of data, default=2017')
    parser.add_argument('-max_item_num', type=int, default=5, help='max number of items, default=5')
    parser.add_argument('-test_cand_num', type=int, default=5, help='number of test candidates, default=5')
    parser.add_argument('-epochs', type=int, default=100, help='number of epochs, default=100')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate, default=0.001')
    parser.add_argument('-patience', type=int, default=5, help='early stopping patience, default=5')
    parser.add_argument('-batch_size', type=int, default=50, help='batch size, default=50')
    parser.add_argument('-rep_vec_num', type=int, default=41, help='number of representative vectors, default=41')
    parser.add_argument('-is_neg_down_sample', type=bool, default=True, help='negative down sampling, default=True')
    parser.add_argument('-pretrained_mlp', type=int, default=0, help='Whether pretrain MLP (not use FC_projection)')
    parser.add_argument('-mlp_projection_dim', type=int, default=128, help='MLP will be learned to achieve designated dimention')
    parser.add_argument("-baseMLPChn", type=int, default=128, help='number of base channel in MLP, default=128')
    parser.add_argument('-train', type=bool, default=True, help='Whether train the model')
    return parser

def parser_comp(): # parser for comp_results.py
    parser = argparse.ArgumentParser(description='MNIST eventotal matching')
    parser.add_argument('-modes', default='3,4', help='list of score modes, maxPooling:0, poolingMA:1, CSS:2, setRepVec_biPMA:3, setRepVec_pivot:4, default:3,4')
    parser.add_argument('-baseChn', type=int, default=64, help='number of base channel, default=32')
    parser.add_argument('-num_layers', default='3', help='list of numbers of layers (attentions) in encoder and decoder, default=3')
    parser.add_argument('-num_heads', default='5', help='list of numbers of heads in attention, default=5')
    parser.add_argument('-is_set_norm', type=int, default=1, help='switch of set-normalization (1:on, 0:off), default=1')
    parser.add_argument('-is_cross_norm', type=int, default=1, help='switch of cross-normalization (1:on, 0:off), default=1')
    parser.add_argument('-trials', default='1,2,3', help='list of indices of trials, default=1,2,3')
    parser.add_argument('-calc_set_sim', type=int, default=0, help='how to evaluate set similarity, CS:0, BERTscore:1, default=0')
    return parser


def load_init_seed_vectors(pickle_path, use_Cvec):
    
    init_seed_pickle_path = os.path.join(pickle_path, "item_seed/item_seed.pkl")

    if not os.path.exists(init_seed_pickle_path):
        print("init_seed vectors haven't been generated ")
        seed_vectors = 0
        rep_vec_num = 0
    else:
        print("Loading init_seed vectors...")
        with open(init_seed_pickle_path, 'rb') as fp:
            seed_vectors = pickle.load(fp)
        
        rep_vec_num = len(seed_vectors)
        seed_vectors = seed_vectors.tolist()

        if not use_Cvec:
            seed_vectors = 0

    return seed_vectors, rep_vec_num


#----------------------------
# plot images in specified sets
def plotImg(imgs,set_IDs,msg="",fname="img_in_sets"):
    _, n_item, _, _, _ = imgs.shape
    n_set = len(set_IDs)
    # fig = plt.figure(figsize=(20,5))
    fig = plt.figure()

    for set_ind in range(n_set):                
        for item_ind in range(n_item):
            fig.add_subplot(n_set, n_item, set_ind*n_item+item_ind+1)
            if item_ind == 0:
                plt.title(f'set:{set_IDs[set_ind]}',fontsize=20)
            if item_ind == 1:
                plt.title(f'{msg}',fontsize=20)

            plt.imshow(imgs[set_IDs[set_ind]][item_ind,:,:,0],cmap="gray")
    
    plt.tight_layout()                
    plt.savefig(f'{fname}.png')
#----------------------------

#----------------------------
# plot loss and accuracy
def plotLossACC(path,loss,val_loss,acc,val_acc):
    epochs = np.arange(len(acc))

    fig=plt.figure()
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    fig.add_subplot(1,2,1)
    plt.plot(epochs,acc,'bo-',label='training acc')
    plt.plot(epochs,val_acc,'b',label='validation acc')
    plt.title('acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    # plt.ylim(0,0.25)
    plt.ylim(0,1)
    
    fig.add_subplot(1,2,2)
    plt.plot(epochs,loss,'bo-',label='training loss')
    plt.plot(epochs,val_loss,'b',label='validation loss')
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(0,3)
    plt.legend()

    path = os.path.join(path,"result/loss_acc.png")
    plt.savefig(path)
#----------------------------

#----------------------------
# plot histogram
def plotHist(corr_pos, corr_neg, mode, fname=''):
    fig = plt.figure(figsize=(20,5))

    max_data_num = np.max([len(corr_neg[0]),len(corr_pos[0])])
    for layer_ind in range(len(corr_pos)):
        fig.add_subplot(1,len(corr_pos),layer_ind+1)
        plt.hist(corr_neg[layer_ind],label="mismatch",bins=np.arange(-1,1.1,0.1))
        plt.hist(corr_pos[layer_ind],alpha=0.5,label="match",bins=np.arange(-1,1.1,0.1))        
        if layer_ind == 0:
            plt.legend(fontsize=12)
        plt.xlim([-1.2,1.2])
        plt.ylim([0,max_data_num])
        plt.xticks(fontsize=12)

        if layer_ind == 0:
            title = 'input'
        elif layer_ind <= (len(corr_pos)-1)/2:
            title = f'enc{layer_ind}'
        else:
            title = f'dec{layer_ind-(len(corr_pos)-1)/2}'

        plt.title(title)
        
    plt.tight_layout()

    if len(fname):
        plt.savefig(fname)
    else:
        plt.show()
#----------------------------

#----------------------------
# function to compute CMC
def calc_cmcs(pred, true_grp, batch_size, qry_ind=0, glry_start_ind=1, top_n=1):

    # reshape predict and true for each batch
    pred_batchs = np.reshape(pred, [-1, batch_size, batch_size])
    true_grp_batchs = np.reshape(true_grp, [-1, batch_size])

    # extract predicted scores for query and compute true labels 
    pred_scores = pred_batchs[:,qry_ind,glry_start_ind:]

    # label
    true_labs = (true_grp_batchs == true_grp_batchs[:,[qry_ind]])[:,glry_start_ind:].astype(int)

    # shuffle pred and true
    np.random.seed(0)
    random_inds = random_inds = np.vstack([np.random.permutation(len(true_labs[0])) for i in range(len(true_labs))]) 
    pred_scores = np.vstack([pred_scores[i][random_inds[i]] for i in range(len(random_inds))])
    true_labs = np.vstack([true_labs[i][random_inds[i]] for i in range(len(random_inds))])

    # sort predicted scores and compute TP map (data x batch_size)
    pred_sort_inds = np.argsort(pred_scores,axis=1)[:,::-1]
    TP_map = np.take_along_axis(true_labs,pred_sort_inds,axis=1)

    cmcs = np.sum(np.cumsum(TP_map,axis=1),axis=0)/len(true_labs)

    return cmcs
#----------------------------

def Set_hinge_loss(scores: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
    """
    Loss function for Set Retrieval task.
    Calculates hinge loss for each set pair in the batch.
    """
    Set_hinge_losssum = []
    slack_variable = 0.2
    batch_size = tf.shape(scores)[0]
    nItem = tf.shape(scores)[1]
    
    for batch_ind in range(batch_size):
        # Get scores for the current set against all others
        score = scores[batch_ind]  # (batch_size,)
        # Get true labels for current set against all others
        y_true_batch = y_true[batch_ind]  # (batch_size,)
        
        # Mask to get positive and negative scores
        positive_score = tf.boolean_mask(score, tf.equal(y_true_batch, 1))  # (num_positives,)
        negative_score = tf.boolean_mask(score, tf.equal(y_true_batch, 0))  # (num_negatives,)
        
        # Debug: Print shapes
        tf.print("Batch:", batch_ind, "Positive scores shape:", tf.shape(positive_score), "Negative scores shape:", tf.shape(negative_score))
        
        # Calculate mean of positive scores
        positive_score = tf.reduce_mean(positive_score)  # scalar
        
        # Calculate hinge loss
        hingeloss = tf.maximum(negative_score - positive_score + slack_variable, 0.0)  # (num_negatives,)
        hingeloss_mean = tf.reduce_mean(hingeloss)  # scalar
        Set_hinge_losssum.append(hingeloss_mean)
    
    Loss = tf.reduce_mean(Set_hinge_losssum)
    return Loss

#----------------------------

def Set_accuracy(score, y_true):
    """Custom Metrics Function to evaluate set similarity between pred item set \hat y and gallery y"""
    """1 : positive_score is in top10 % of set similarity pairs , 0 : otherwise"""
    # threshold K 
    threk = int(len(score)*0.01)
    
    accuracy = np.zeros((len(score), 1))

    for batch_ind in range(len(score)):
        f1_score = score[batch_ind]
        _, topscore_index = tf.nn.top_k(f1_score, k=threk)
        if tf.where(tf.equal(y_true[batch_ind], 1))[0].numpy() in topscore_index: # (tf.where(tf.equal(y_true[batch_ind], 1))[0].numpy()) finds positive index.
            accuracy[batch_ind] += 1

    return accuracy

#----------------------------


def plot_3d_tsne(seed_vectors, output_filename="seed_vectors_tsne.png"):
    # リストから NumPy 配列に変換
    seed_vectors = np.array(seed_vectors)

    # t-SNE のインスタンス作成
    tsne = TSNE(n_components=3, perplexity=30, random_state=0)
    seed_vectors_reduced = tsne.fit_transform(seed_vectors)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # カテゴリIDの頭2桁に基づいた色の辞書を作成
    category_colors = {
        '10': 'red',    # アウター
        '11': 'green',  # インナー
        '12': 'blue',   # ボトムス
        '13': 'cyan',   # 靴
        '14': 'magenta', # 帽子
        '15': 'yellow', # アクセサリー
        '16': 'orange'  # 帽子
    }

    # カテゴリIDリスト
    category_ids = ["10001", "10002", "10003", "10004", "10005", "11001", "11002", "11003", "11004", "11005", "11006", "11007", "12001", "12002", "12003", "12004", "12005", "13001", "13002", "13003", "13004", "13005", "14001", "14002", "14003", "14004", "14005", "14006", "14007", "15001", "15002", "15003", "15004", "15005", "15006", "15007", "16001", "16002", "16003", "16004", "16005"]

    # データ点をプロットし、各点にカテゴリIDをラベルとして追加
    for i, vec in enumerate(seed_vectors_reduced):
        category_id = category_ids[i]
        head_two = category_id[:2]
        color = category_colors.get(head_two, 'gray')  # カテゴリIDの頭2桁に基づく色
        ax.scatter(vec[0], vec[1], vec[2], color=color)
        ax.text(vec[0], vec[1], vec[2], f'{category_id}', color=color)

    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_title('3D t-SNE of Seed Vectors with Category IDs')
    plt.savefig("seed_vectors_tsne.png")