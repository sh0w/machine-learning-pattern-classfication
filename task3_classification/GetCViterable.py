#use as GetCViterable().iterable()

import numpy as np
import pandas as pd
import copy

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit


class GetCViterable():
    def iterable(self,path="dataset/phase_3_TRAIN_7d499bff69ca69b6_6372c3e_MLPC2021_generic.csv", pianist_folds=3,pieces_folds=5,doPianists=True,doPieces=True):
        df = pd.read_csv(path)

        #create tags_dataframe:
        X_tags=pd.DataFrame()
        X_tags['id']=df['id']


        #extract piece_id and pianist to later allow by piece/pianist/both cross validation
        def extractPianist(x):
            return x[0:2]
        def extract_piece_id(x):
            return x[3:5]
        def extract_snippet_number(x):
            return x[6:9]

        #get relevant information from dataframe
        X_tags['Pianist']=X_tags['id'].apply(extractPianist)
        X_tags['Piece_id']=X_tags['id'].apply(extract_piece_id)
        X_tags['Snippet_number']=X_tags['id'].apply(extract_snippet_number)
        X_tags['class']=df['quadrant']

        #get list of pianists and pieces!
        pianist_list=list(set(X_tags['Pianist']))
        piece_list=list(set(X_tags['Piece_id']))

        #choose random pianists
        pianist_cv=[]
        number_folds=pianist_folds

        #initialize pianists_cv:
        for i in range(number_folds):
            pianist_cv.append([])

        #assign pianists to folds randomly
        cp_pianist_list=pianist_list.copy()
        for i in range(len(cp_pianist_list)):
            choosen_pianist=cp_pianist_list[np.random.randint(len(cp_pianist_list), size=1)[0]]
            cp_pianist_list.remove(choosen_pianist)
            pianist_cv[len(cp_pianist_list)%number_folds].append(choosen_pianist)

        #get distribution of classes per piece
        piece_dist_list=[]
        piece_dist_list_perecent=[]
        for piece in piece_list:
            mylist=[]
            for i in range(4):
                mylist.append(len(X_tags.loc[(X_tags['Piece_id'] == piece) & (X_tags['class']==i+1)]))
            piece_dist_list.append(mylist)
            mylist=[element/sum(mylist) for element in mylist]
            piece_dist_list_perecent.append(mylist)

        #create 4 subsets:
        piece_subsets=[]

        #initialize piece_cv:
        for i in range(4):
            piece_subsets.append([])
        piece_list_cp=copy.deepcopy(piece_list)
        for p,piece in enumerate(piece_list):
            done=False
            for i in range(4):
                if not done:
                    #print(piece_dist_list[p][i])
                    #print(max(piece_dist_list[p]))
                    if piece_dist_list[p][i]==max(piece_dist_list[p][:]):
                        piece_subsets[i].append(piece_list[p])
                        piece_list_cp.remove
                        done=True

        # chose random pieces using above subset technique:
        piece_cv=[]
        number_folds=pieces_folds
        #initialize piece_cv:
        for i in range(number_folds):
            piece_cv.append([])

        cp_pi_subs = copy.deepcopy(piece_subsets)
        #assign pieces to folds randomly
        for i in range(4):
            fold=0
            for p in range(len(piece_subsets[i])):
                chos_piece=cp_pi_subs[i][np.random.randint(len(cp_pi_subs[i]), size=1)[0]]
                cp_pi_subs[i].remove(chos_piece)
                piece_cv[fold].append(chos_piece)
                if fold>=number_folds-1:
                    fold=0
                else:
                    fold+=1
        #creating masks for gridsearch cv:

        #create vertical splits:
        myCViterator = []

        if(doPianists):
            for f,fold in enumerate(pianist_cv):
                vetrical_mask_train=np.zeros(0).astype(int)
                vetrical_mask_test=np.zeros(0).astype(int)
                for i in range(len(fold)):
                    mask= X_tags[X_tags['Pianist']!=pianist_cv[f][i]].index.values.astype(int)
                    vetrical_mask_train=np.concatenate((vetrical_mask_train,mask), axis=0)

                    mask= X_tags[X_tags['Pianist']==pianist_cv[f][i]].index.values.astype(int)
                    vetrical_mask_test=np.concatenate((vetrical_mask_test,mask), axis=0)
                #print(vetrical_mask_test)
                #print(vetrical_mask_train)
                trainIndices=vetrical_mask_train
                testIndices=vetrical_mask_test
                myCViterator.append((trainIndices, testIndices))

        if (doPieces):
            #and create horizontal splits:
            for f,fold in enumerate(piece_cv):
                vetrical_mask_train=np.zeros(0).astype(int)
                vetrical_mask_test=np.zeros(0).astype(int)
                for i in range(len(fold)):
                    mask= X_tags[X_tags['Piece_id']!=piece_cv[f][i]].index.values.astype(int)
                    vetrical_mask_train=np.concatenate((vetrical_mask_train,mask), axis=0)

                    mask= X_tags[X_tags['Piece_id']==piece_cv[f][i]].index.values.astype(int)
                    vetrical_mask_test=np.concatenate((vetrical_mask_test,mask), axis=0)
                #print(vetrical_mask_test)
                #print(vetrical_mask_train)
                trainIndices=vetrical_mask_train
                testIndices=vetrical_mask_test
                myCViterator.append((trainIndices, testIndices))

        return myCViterator