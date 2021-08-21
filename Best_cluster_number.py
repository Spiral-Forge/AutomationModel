# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 21:09:54 2021

@author: godwi
"""

import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import numpy as np

import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA


def Mentor_training(Mentor_json,file_location=''):
    
    def data_preprocessing(data):
        data=data.drop(labels=['branch','id','name','rollNo','peerID','post','hosteller'],axis=1)
        def cleaning_language_data(values):
            v=str(values)
            v=v.strip()
            s=re.sub(pattern="[^\w\s\,\/\+]",repl="",string=v)
            s=s.lower()
            res = re.split(', |/', s)
            count=0
#     for i in res:
#         res[count]=i.replace('c++','chigh')
#         res[count]=i.replace('c','clow')
#         res[count]=i.replace('c#','cmedium')
        
#         count+=1
            x=' '.join(res)
            x=x.replace('c++','chigh')
            x=x.replace('c','clow')
            x=x.replace('c#','cmedium')
            x=x.replace('clowhigh','chigh')
            x=x.replace('clow#','cmedium')
            x=x.replace('no preferenclowe','no preference')
            return x

        def cleaning_domain_data(values):
            v=str(values)
            v=v.strip()
            s=re.sub(pattern="[^\w\s\,\/\+]",repl="",string=v)
            s=s.lower()
            res = re.split(', |/', s)
            count=0
#     for i in res:
#         res[count]=i.replace('c++','chigh')
#         res[count]=i.replace('c','clow')
#         res[count]=i.replace('c#','cmedium')
        
#         count+=1
            x=' '.join(res)
            x=x.replace('no prefence','no preference')
    
            return x
    
    
        for name,values in data.items():
            if name=='domains':
                if pd.api.types.is_string_dtype(values):
                    count=0
                    for i in values:
                        data.iloc[count,0]=cleaning_domain_data(i)
                        count+=1
            elif name=='languages':
                if pd.api.types.is_string_dtype(values):
                    count=0
                    for i in values:
                        data.iloc[count,1]=cleaning_language_data(i)
                        count+=1
    
        return data
    
    def find_best_cluster(X):
        sil_avg=[]
        best_cluster=[]
        range_clusters=[2,3,4,5,6,7,8,9]

        for n_clusters in range_clusters:
    
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)

    
            silhouette_avg = silhouette_score(X, cluster_labels)
            sample_silhouette_values = silhouette_samples(X, cluster_labels)
            sample_silhouette_values.sort()
            if sample_silhouette_values[0]>0:
                sil_avg.append(silhouette_avg)
                best_cluster.append(n_clusters)
                max_score=np.argmax(sil_avg)  

        return best_cluster[max_score]
    
    
    def cluster_model(Mentor_json):
        Mentor_df_json=pd.read_json(Mentor_json)
        Mentor_df=pd.DataFrame(Mentor_df_json.users.values.tolist())
        data=Mentor_df.copy()
        df=pd.DataFrame({'domains':['[Web Development, App Development, Machine Learning, IOT, BlockChain, AR/VR, Game Development, Cloud Engineering, Competitive Programming, Cyber Security, Open Source]'],'languages':['[Java, Python, C/C++, No Preference]']})
        data=data.append(df,ignore_index=True)
        mentor_df_pre=data_preprocessing(data)

        np.random.seed(10)
        domains_vector=CountVectorizer()
        domains_vec_val=domains_vector.fit_transform(mentor_df_pre['domains'])
        df_domain_wrds=pd.DataFrame(domains_vec_val.toarray(),columns=domains_vector.get_feature_names())
        np.random.seed(11)
        languages_vector=CountVectorizer()
        languages_vec_val=languages_vector.fit_transform(mentor_df_pre['languages'])
        df_lang_wrds=pd.DataFrame(languages_vec_val.toarray(),columns=languages_vector.get_feature_names())
        final_df=pd.concat([df_domain_wrds,df_lang_wrds],axis=1)
        final_df=final_df.drop(index=final_df.shape[0]-1,axis=0)
        pca=PCA(n_components=2,random_state=30)
        X=pca.fit_transform(final_df)
        cluster_num=find_best_cluster(X)
        model=KMeans(n_clusters=cluster_num, random_state=25)
        cluster_labels = model.fit_predict(X)
        
        
    return pickle.dump(model,open(file_location+'cluster_on_trained_mentor.pkl','wb')) 